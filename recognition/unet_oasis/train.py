"""
Task 2 UNet —— 训练脚本

目标: 所有标签的 DSC > 0.9（验证/测试集）
运行: python recognition/unet_oasis/train.py --epochs 20
产物: checkpoints/unet.pth、outputs/loss_curve.png、outputs/dice_curve.png
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import DEFAULT_ROOT, NUM_CLASSES, get_dataloaders, to_one_hot
from modules import UNet, CombinedLoss, dice_from_counts, hard_dice_counts

HERE = Path(__file__).parent


def pick_device() -> torch.device:
    """优先 CUDA（Rangpur A100），其次 Apple MPS，最后退回 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_workers(requested: int, device: torch.device) -> int:
    """决定 DataLoader 的 worker 数量。

    实测（M5 Mac，OASIS 256x256）：单张图解码只要 ~1 ms，而多进程 worker
    的 IPC 序列化开销远大于此 —— num_workers=0 是 9 ms/batch，
    num_workers=6 反而要 262 ms/batch，慢 29 倍；且 fork 与 MPS 并存时
    还可能把主进程卡在不可中断等待上。
    所以：CUDA（集群，CPU 核多、数据在网络盘）用多 worker，其余一律 0。
    """
    if requested >= 0:
        return requested
    return 4 if device.type == "cuda" else 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=-1,
                   help="-1 表示按设备自动选择（CUDA 用 4，MPS/CPU 用 0）")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=50,
                   help="每多少个 step 打印一次进度")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device):
    """整个验证集上的逐类硬 DSC，形状 (C,)。

    交集与基数在全集上累加后再相除，得到数据集级别的 DSC
    （而不是各 batch DSC 的平均值 —— 两者并不相等）。
    """
    model.eval()
    inter = torch.zeros(NUM_CLASSES, device=device)
    card = torch.zeros(NUM_CLASSES, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        i, c = hard_dice_counts(model(x), to_one_hot(y, NUM_CLASSES))
        inter += i
        card += c
    return dice_from_counts(inter, card).cpu()


def main():
    args = parse_args()
    device = pick_device()
    use_amp = (not args.no_amp) and device.type == "cuda"
    (HERE / "checkpoints").mkdir(exist_ok=True)
    (HERE / "outputs").mkdir(exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        args.data_root, args.batch_size, args.image_size, auto_workers(args.num_workers, device))

    model = UNet(1, NUM_CLASSES, args.base).to(device)
    criterion = CombinedLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    losses, dices = [], []
    best = 0.0
    n_steps = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        epoch_start = time.perf_counter()
        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y, to_one_hot(y, NUM_CLASSES))
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += loss.item()

            # 定期打印步进度：一个 epoch 在 MPS 上要好几分钟，
            # 没有中间输出的话根本分不清「在算」和「卡死」。
            # flush=True 是必须的 —— 重定向到文件时 stdout 是块缓冲的。
            if step % args.log_every == 0 or step == n_steps:
                elapsed = time.perf_counter() - epoch_start
                eta = elapsed / step * (n_steps - step)
                print(f"  epoch {epoch:3d} [{step:4d}/{n_steps}] "
                      f"loss {running / step:.4f} | {elapsed / step * 1000:.0f} ms/step "
                      f"| 本轮剩余 ~{eta / 60:.1f} min", flush=True)
        scheduler.step()

        dsc = validate(model, val_loader, device)
        losses.append(running / len(train_loader))
        dices.append(dsc.tolist())
        per_class = " ".join(f"c{i}:{d:.4f}" for i, d in enumerate(dsc))
        print(f"epoch {epoch:3d} | loss {losses[-1]:.4f} | DSC {per_class} "
              f"| mean {dsc.mean():.4f} | {(time.perf_counter() - epoch_start) / 60:.1f} min",
              flush=True)

        if dsc.mean() > best:
            best = dsc.mean().item()
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": epoch, "dsc": dsc.tolist()},
                       HERE / "checkpoints" / "unet.pth")

    plt.figure(); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True)
    plt.savefig(HERE / "outputs" / "loss_curve.png", dpi=120)

    plt.figure()
    dices_t = torch.tensor(dices)
    for c in range(NUM_CLASSES):
        plt.plot(dices_t[:, c], label=f"class {c}")
    plt.axhline(0.9, ls="--", c="k", label="target 0.9")
    plt.xlabel("epoch"); plt.ylabel("DSC"); plt.legend(); plt.grid(True)
    plt.savefig(HERE / "outputs" / "dice_curve.png", dpi=120)

    final = torch.tensor(dices[-1])
    print(f"\n最佳平均 DSC {best:.4f}")
    print("全部标签达标 (>0.9)" if (final > 0.9).all() else "仍有标签未达 0.9，需要继续训练/调参")


if __name__ == "__main__":
    main()
