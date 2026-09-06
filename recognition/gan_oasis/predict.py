"""
Task 3 GAN —— generation and evaluation (this is what runs live in the demo)

Run:
  python recognition/gan_oasis/predict.py --n 64
Outputs:
  outputs/generated_samples.png    grid of generated brain images
  outputs/interpolation.png        linear interpolation between two latent vectors
  printed comparison of the diversity score against the real data

The interpolation figure is the direct visual evidence on mode collapse: a collapsed
generator jumps between a few modes along the path, whereas a healthy one transitions
smoothly.
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
    """Load the generator. EMA weights by default: their sample quality is clearly steadier."""
    state = torch.load(ckpt_path, map_location=device)
    saved = state.get("args", {})
    gen = Generator(saved.get("latent_dim", 128), saved.get("base", 64),
                    saved.get("image_size", 128)).to(device)
    key = "generator_ema" if (use_ema and "generator_ema" in state) else "generator"
    gen.load_state_dict(state[key])
    gen.eval()
    print(f"Loaded checkpoint: epoch {state['epoch']}, using the {key} weights")
    return gen, saved


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(HERE / "checkpoints" / "gan.pth"))
    p.add_argument("--data-root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--n", type=int, default=64, help="how many images to generate")
    p.add_argument("--no-ema", action="store_true", help="use raw weights instead of EMA weights")
    p.add_argument("--interp-steps", type=int, default=10)
    args = p.parse_args()

    device = pick_device()
    gen, saved = load_generator(args.ckpt, device, use_ema=not args.no_ema)
    outdir = HERE / "outputs"; outdir.mkdir(exist_ok=True)

    # ---- grid of generated samples ----
    z = torch.randn(args.n, gen.latent_dim, device=device)
    samples = gen(z)
    save_image(samples, outdir / "generated_samples.png", nrow=8,
               normalize=True, value_range=(-1, 1))
    print("Saved generated samples:", outdir / "generated_samples.png")

    # ---- Diversity comparison: the objective basis for the "no mode collapse" claim ----
    loader = get_dataloader(args.data_root, args.n,
                            saved.get("image_size", 128), 0, split="test")
    real = next(iter(loader)).to(device)
    div_fake = diversity_score(samples)
    div_real = diversity_score(real)
    print(f"\n--- diversity (mean pairwise L2 distance between samples) ---")
    print(f"  generated: {div_fake:.2f}")
    print(f"  real     : {div_real:.2f}")
    print(f"  ratio    : {div_fake / div_real * 100:.0f}%")
    print("  verdict  : " + ("close to the real data, no obvious mode collapse"
                             if div_fake / div_real > 0.7
                             else "clearly low, mode collapse present"))

    # ---- Latent-space interpolation: a collapsed generator jumps part-way along ----
    z_a = torch.randn(8, gen.latent_dim, device=device)
    z_b = torch.randn(8, gen.latent_dim, device=device)
    alphas = torch.linspace(0, 1, args.interp_steps, device=device)
    # each row interpolates between one pair of endpoints, 8 rows in total
    rows = [gen(torch.lerp(z_a, z_b, a.item())) for a in alphas]
    grid = torch.stack(rows, dim=1).flatten(0, 1)     # (8, steps, 1,H,W) -> (8*steps,...)
    save_image(grid, outdir / "interpolation.png", nrow=args.interp_steps,
               normalize=True, value_range=(-1, 1))
    print("Saved latent-space interpolation:", outdir / "interpolation.png")


if __name__ == "__main__":
    main()
