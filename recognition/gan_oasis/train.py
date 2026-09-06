"""
Task 3 GAN —— training script

Run:
  python recognition/gan_oasis/train.py --epochs 60
Outputs:
  checkpoints/gan.pth
  outputs/samples_epoch{NN}.png   samples during training (fixed noise, shows the evolution)
  outputs/loss_curve.png          discriminator / generator losses
  outputs/diversity_curve.png     diversity score vs real-data baseline (evidence on mode collapse)

The task sheet warns about mode collapse, so four countermeasures are used here and the
**diversity score is plotted**, which makes "did it collapse" a conclusion backed by numbers
rather than by eyeballing the sample grids.
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
    """Prefer CUDA (Rangpur A100), then Apple MPS, and fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_workers(requested: int, device: torch.device) -> int:
    """Multiple workers on CUDA, 0 on MPS/CPU (see the measurements in unet_oasis/train.py)."""
    return requested if requested >= 0 else (4 if device.type == "cuda" else 0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--base", type=int, default=64)
    # TTUR: discriminator learning rate slightly above the generator's, repeatedly shown to help
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=3e-4)
    p.add_argument("--r1-gamma", type=float, default=10.0, help="R1 gradient penalty strength")
    p.add_argument("--r1-every", type=int, default=16,
                   help="apply R1 once every N steps (lazy regularisation)")
    p.add_argument("--label-smooth", type=float, default=0.9,
                   help="real labels are 0.9 rather than 1.0")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=5, help="save a sample grid every N epochs")
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

    # betas=(0.5, 0.999) is the standard DCGAN setting: lower the first-order momentum,
    # because a GAN chases a target that keeps moving and too much momentum lags behind it
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(64, args.latent_dim, device=device)   # fixed noise, shows the evolution
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

            # ---------------- discriminator ----------------
            z = torch.randn(batch, args.latent_dim, device=device)
            fake = gen(z).detach()          # detach: no generator update in this step
            logits_real = disc(real)
            logits_fake = disc(fake)
            # Real labels are 0.9 (label smoothing): stop the discriminator pushing its
            # confidence into saturation, where the gradient is ~0 and the generator learns nothing
            loss_d = (bce(logits_real, torch.full_like(logits_real, args.label_smooth))
                      + bce(logits_fake, torch.zeros_like(logits_fake)))

            # Lazy R1 regularisation: applied every r1_every steps, as the gradient penalty
            # needs second-order derivatives and is costly. The r1_every factor keeps the
            # average strength the same as applying it at every step.
            if step % args.r1_every == 0:
                penalty = r1_penalty(disc, real)
                loss_d = loss_d + (args.r1_gamma / 2) * penalty * args.r1_every

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---------------- generator ----------------
            z = torch.randn(batch, args.latent_dim, device=device)
            fake = gen(z)
            # Non-saturating loss: maximise log D(G(z)), not minimise log(1 - D(G(z))), whose
            # gradient nearly vanishes early on while D is strong, as the original GAN paper noted.
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

        # ---------------- end of epoch: sampling + diversity ----------------
        gen.eval()
        with torch.no_grad():
            samples = ema.shadow(torch.randn(64, args.latent_dim, device=device))
        div_fake = diversity_score(samples)
        div_real = diversity_score(real)          # baseline: last real batch of this epoch

        history["d"].append(sum_d / n_steps)
        history["g"].append(sum_g / n_steps)
        history["div_fake"].append(div_fake)
        history["div_real"].append(div_real)
        print(f"epoch {epoch:3d} | D {history['d'][-1]:.4f} | G {history['g'][-1]:.4f} "
              f"| diversity gen {div_fake:.2f} / real {div_real:.2f} "
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

    # ---------------- curves ----------------
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
    print(f"\nTraining finished in {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"Final diversity: generated {history['div_fake'][-1]:.2f} / "
          f"real {history['div_real'][-1]:.2f} = {ratio * 100:.0f}%")
    print("Diversity close to the real data, no obvious mode collapse" if ratio > 0.7
          else "Diversity clearly below the real data: mode collapse, retuning needed")


if __name__ == "__main__":
    main()
