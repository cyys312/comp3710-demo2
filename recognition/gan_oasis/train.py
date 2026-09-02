"""
Task 3 GAN —— 训练脚本

运行:
  python recognition/gan_oasis/train.py --epochs 60
产物:
  checkpoints/gan.pth
  outputs/samples_epoch{NN}.png   训练过程中的生成样本（固定噪声，便于看演化）
  outputs/loss_curve.png          判别器/生成器损失
  outputs/diversity_curve.png     多样性指标 vs 真实数据基准（mode collapse 的证据）

针对任务书警告的 mode collapse，这里用了四项措施，并且**把多样性指标画出来**，
使「有没有崩塌」成为可以摆出数字的结论，而不是靠肉眼看图。
"""

import argparse
import time
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

from dataset import DEFAULT_ROOT, get_dataloader  # noqa: E402
from modules import (Discriminator, EMA, Generator, diversity_score,  # noqa: E402
                     init_weights, r1_penalty)

HERE = Path(__file__).parent


def pick_device() -> torch.device:
    """优先 CUDA（Rangpur A100），其次 Apple MPS，最后退回 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_workers(requested: int, device: torch.device) -> int:
    """CUDA 用多 worker，MPS/CPU 用 0（见 unet_oasis/train.py 的实测说明）。"""
    return requested if requested >= 0 else (4 if device.type == "cuda" else 0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--base", type=int, default=64)
    # TTUR：判别器学习率略高于生成器，是 GAN 训练里被反复验证有效的设置
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=3e-4)
    p.add_argument("--r1-gamma", type=float, default=10.0, help="R1 梯度惩罚强度")
    p.add_argument("--r1-every", type=int, default=16, help="每多少步做一次 R1（惰性正则）")
    p.add_argument("--label-smooth", type=float, default=0.9, help="真实标签用 0.9 而非 1.0")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=5, help="每多少个 epoch 存一次样本图")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    (HERE / "checkpoints").mkdir(exist_ok=True)
    (HERE / "outputs").mkdir(exist_ok=True)
    print(f"device={device} | image_size={args.image_size} | latent_dim={args.latent_dim}",
          flush=True)

    loader = get_dataloader(args.data_root, args.batch_size, args.image_size,
                            auto_workers(args.num_workers, device))

    gen = Generator(args.latent_dim, args.base, args.image_size).to(device)
    disc = Discriminator(args.base, args.image_size).to(device)
    gen.apply(init_weights)
    disc.apply(init_weights)
    ema = EMA(gen, args.ema_decay)

    # betas=(0.5, 0.999) 是 DCGAN 的标准设置：降低一阶动量，
    # 因为 GAN 的目标是个不断移动的靶子，动量太大反而拖累追踪
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(64, args.latent_dim, device=device)   # 固定噪声，看演化
    n_steps = len(loader)
    history = {"d": [], "g": [], "div_fake": [], "div_real": []}
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        gen.train(); disc.train()
        sum_d = sum_g = 0.0
        epoch_start = time.perf_counter()

        for step, real in enumerate(loader, start=1):
            real = real.to(device, non_blocking=True)
            batch = real.size(0)

            # ---------------- 判别器 ----------------
            z = torch.randn(batch, args.latent_dim, device=device)
            fake = gen(z).detach()          # detach：这一步不更新生成器
            logits_real = disc(real)
            logits_fake = disc(fake)
            # 真实标签用 0.9（label smoothing）：不让判别器把置信度推到饱和区，
            # 饱和之后梯度接近 0，生成器就学不到东西了
            loss_d = (bce(logits_real, torch.full_like(logits_real, args.label_smooth))
                      + bce(logits_fake, torch.zeros_like(logits_fake)))

            # 惰性 R1 正则：每 r1_every 步做一次，梯度惩罚要算二阶导，比较贵。
            # 乘以 r1_every 是为了让平均强度与每步都做时保持一致。
            if step % args.r1_every == 0:
                penalty = r1_penalty(disc, real)
                loss_d = loss_d + (args.r1_gamma / 2) * penalty * args.r1_every

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---------------- 生成器 ----------------
            z = torch.randn(batch, args.latent_dim, device=device)
            fake = gen(z)
            # 非饱和损失：最大化 log D(G(z))，而不是最小化 log(1 - D(G(z)))。
            # 后者在训练初期判别器很强时梯度几乎为 0，是原始 GAN 论文就指出的问题。
            loss_g = bce(disc(fake), torch.ones(batch, device=device))

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            ema.update(gen)

            sum_d += loss_d.item()
            sum_g += loss_g.item()
            if step % args.log_every == 0 or step == n_steps:
                elapsed = time.perf_counter() - epoch_start
                print(f"  epoch {epoch:3d} [{step:4d}/{n_steps}] "
                      f"D {sum_d / step:.4f} | G {sum_g / step:.4f} "
                      f"| {elapsed / step * 1000:.0f} ms/step", flush=True)

        # ---------------- 每轮结束：采样 + 多样性 ----------------
        gen.eval()
        with torch.no_grad():
            samples = ema.shadow(torch.randn(64, args.latent_dim, device=device))
        div_fake = diversity_score(samples)
        div_real = diversity_score(real)          # 用本轮最后一个真实 batch 作基准

        history["d"].append(sum_d / n_steps)
        history["g"].append(sum_g / n_steps)
        history["div_fake"].append(div_fake)
        history["div_real"].append(div_real)
        print(f"epoch {epoch:3d} | D {history['d'][-1]:.4f} | G {history['g'][-1]:.4f} "
              f"| 多样性 生成 {div_fake:.2f} / 真实 {div_real:.2f} "
              f"({div_fake / div_real * 100:.0f}%) "
              f"| {time.perf_counter() - epoch_start:.0f}s", flush=True)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                grid = ema.shadow(fixed_z)
            save_image(grid, HERE / "outputs" / f"samples_epoch{epoch:03d}.png",
                       nrow=8, normalize=True, value_range=(-1, 1))
            torch.save({"generator": gen.state_dict(),
                        "generator_ema": ema.shadow.state_dict(),
                        "discriminator": disc.state_dict(),
                        "args": vars(args), "epoch": epoch},
                       HERE / "checkpoints" / "gan.pth")

    # ---------------- 曲线 ----------------
    plt.figure()
    plt.plot(history["d"], label="Discriminator")
    plt.plot(history["g"], label="Generator")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True)
    plt.title("GAN training losses")
    plt.savefig(HERE / "outputs" / "loss_curve.png", dpi=120)

    plt.figure()
    plt.plot(history["div_fake"], label="Generated")
    plt.plot(history["div_real"], label="Real (reference)", ls="--")
    plt.xlabel("epoch"); plt.ylabel("mean pairwise L2 distance")
    plt.title("Sample diversity: quantitative check for mode collapse")
    plt.legend(); plt.grid(True)
    plt.savefig(HERE / "outputs" / "diversity_curve.png", dpi=120)

    ratio = history["div_fake"][-1] / history["div_real"][-1]
    print(f"\n训练完成，共 {(time.perf_counter() - t0) / 60:.1f} 分钟")
    print(f"最终多样性：生成 {history['div_fake'][-1]:.2f} / "
          f"真实 {history['div_real'][-1]:.2f} = {ratio * 100:.0f}%")
    print("多样性接近真实数据，无明显 mode collapse" if ratio > 0.7
          else "多样性明显低于真实数据，存在 mode collapse，需要调参")


if __name__ == "__main__":
    main()
