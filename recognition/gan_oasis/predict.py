"""
Task 3 GAN —— 生成与评估（demo 现场跑这个）

运行:
  python recognition/gan_oasis/predict.py --n 64
产物:
  outputs/generated_samples.png    生成的脑图网格
  outputs/interpolation.png        两个隐向量之间的线性插值
  多样性指标与真实数据的对比打印

插值那张图是判断 mode collapse 的直观证据：若生成器崩塌，
沿插值路径的图像会在几个模式之间突变；健康的生成器应当平滑过渡。
"""

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
from torchvision.utils import save_image  # noqa: E402

from dataset import DEFAULT_ROOT, get_dataloader  # noqa: E402
from modules import Generator, diversity_score  # noqa: E402

HERE = Path(__file__).parent


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_generator(ckpt_path, device, use_ema: bool = True):
    """载入生成器。默认用 EMA 权重——它的样本质量明显更稳。"""
    state = torch.load(ckpt_path, map_location=device)
    saved = state.get("args", {})
    gen = Generator(saved.get("latent_dim", 128), saved.get("base", 64),
                    saved.get("image_size", 128)).to(device)
    key = "generator_ema" if (use_ema and "generator_ema" in state) else "generator"
    gen.load_state_dict(state[key])
    gen.eval()
    print(f"载入 checkpoint: epoch {state['epoch']}，使用 {key} 权重")
    return gen, saved


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(HERE / "checkpoints" / "gan.pth"))
    p.add_argument("--data-root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--n", type=int, default=64, help="生成多少张")
    p.add_argument("--no-ema", action="store_true", help="用原始权重而非 EMA 权重")
    p.add_argument("--interp-steps", type=int, default=10)
    args = p.parse_args()

    device = pick_device()
    gen, saved = load_generator(args.ckpt, device, use_ema=not args.no_ema)
    outdir = HERE / "outputs"; outdir.mkdir(exist_ok=True)

    # ---- 生成样本网格 ----
    z = torch.randn(args.n, gen.latent_dim, device=device)
    samples = gen(z)
    save_image(samples, outdir / "generated_samples.png", nrow=8,
               normalize=True, value_range=(-1, 1))
    print("已保存生成样本:", outdir / "generated_samples.png")

    # ---- 多样性对比：这是「无 mode collapse」的客观依据 ----
    loader = get_dataloader(args.data_root, args.n,
                            saved.get("image_size", 128), 0, split="test")
    real = next(iter(loader)).to(device)
    div_fake = diversity_score(samples)
    div_real = diversity_score(real)
    print(f"\n--- 多样性（样本两两平均 L2 距离）---")
    print(f"  生成样本: {div_fake:.2f}")
    print(f"  真实样本: {div_real:.2f}")
    print(f"  比值    : {div_fake / div_real * 100:.0f}%")
    print("  判定    : " + ("接近真实数据，无明显 mode collapse"
                            if div_fake / div_real > 0.7 else "明显偏低，存在 mode collapse"))

    # ---- 隐空间插值：崩塌的生成器会在途中突变 ----
    z_a = torch.randn(8, gen.latent_dim, device=device)
    z_b = torch.randn(8, gen.latent_dim, device=device)
    alphas = torch.linspace(0, 1, args.interp_steps, device=device)
    # 每一行是一对端点之间的插值，共 8 行
    rows = [gen(torch.lerp(z_a, z_b, a.item())) for a in alphas]
    grid = torch.stack(rows, dim=1).flatten(0, 1)     # (8, steps, 1,H,W) -> (8*steps,...)
    save_image(grid, outdir / "interpolation.png", nrow=args.interp_steps,
               normalize=True, value_range=(-1, 1))
    print("已保存隐空间插值:", outdir / "interpolation.png")


if __name__ == "__main__":
    main()
