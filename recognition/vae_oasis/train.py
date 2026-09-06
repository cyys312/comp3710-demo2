"""
Task 1 VAE — training script

Run: python recognition/vae_oasis/train.py --epochs 30 --latent-dim 32
Produces: checkpoints/vae.pth, outputs/loss_curve.png, outputs/recon_epochXX.png
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from dataset import DEFAULT_ROOT, get_dataloaders
from modules import VAE, vae_loss

HERE = Path(__file__).parent


def pick_device() -> torch.device:
    """Prefer CUDA (Rangpur A100), then Apple MPS, falling back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_workers(requested: int, device: torch.device) -> int:
    """Decide how many DataLoader workers to use.

    Measured (M5 Mac, OASIS 256x256): decoding one image costs only ~1 ms, while the IPC
    serialisation overhead of multiprocessing workers dwarfs that — num_workers=0 gives
    9 ms/batch, num_workers=6 gives 262 ms/batch, 29x slower; fork alongside MPS can also
    leave the main process wedged in an uninterruptible wait.
    So: several workers on CUDA (cluster, many cores, data on network storage), 0 elsewhere.
    """
    if requested >= 0:
        return requested
    return 4 if device.type == "cuda" else 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=-1,
                   help="-1 picks automatically per device (4 on CUDA, 0 on MPS/CPU)")
    return p.parse_args()


def run_epoch(model, loader, device, beta, optimiser=None):
    train = optimiser is not None
    model.train() if train else model.eval()
    totals = [0.0, 0.0, 0.0]
    with torch.set_grad_enabled(train):
        for x in loader:
            x = x.to(device, non_blocking=True)
            recon, mu, logvar = model(x)
            loss, rec, kl = vae_loss(recon, x, mu, logvar, beta)
            if train:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
            for i, v in enumerate((loss, rec, kl)):
                totals[i] += v.item()
    return [t / len(loader) for t in totals]


def main():
    args = parse_args()
    device = pick_device()
    (HERE / "checkpoints").mkdir(exist_ok=True)
    (HERE / "outputs").mkdir(exist_ok=True)
    # Checkpoints and figures are named by latent dimensionality: latent_dim=2 (2D manifold grid)
    # and latent_dim=32 (sharper reconstructions) are trained side by side and must not collide.
    tag = f"z{args.latent_dim}"
    print(f"device={device} | latent_dim={args.latent_dim} | tag={tag}", flush=True)

    train_loader, val_loader, _ = get_dataloaders(
        args.data_root, args.batch_size, args.image_size, auto_workers(args.num_workers, device))
    model = VAE(args.latent_dim, image_size=args.image_size).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train": [], "val": []}
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        tr = run_epoch(model, train_loader, device, args.beta, optimiser)
        va = run_epoch(model, val_loader, device, args.beta)
        history["train"].append(tr[0])
        history["val"].append(va[0])
        print(f"epoch {epoch:3d} | train {tr[0]:.2f} (rec {tr[1]:.2f} kl {tr[2]:.2f}) "
              f"| val {va[0]:.2f} | {time.perf_counter() - epoch_start:.0f}s", flush=True)

        if va[0] < best:
            best = va[0]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                       HERE / "checkpoints" / f"vae_{tag}.pth")

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                x = next(iter(val_loader))[:8].to(device)
                recon, _, _ = model(x)
                save_image(torch.cat([x, recon]),
                           HERE / "outputs" / f"{tag}_recon_epoch{epoch:02d}.png", nrow=8)

    plt.figure()
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("epoch"); plt.ylabel("ELBO loss"); plt.legend(); plt.grid(True)
    plt.savefig(HERE / "outputs" / f"{tag}_loss_curve.png", dpi=120)
    print(f"best validation loss {best:.2f}, checkpoint saved")


if __name__ == "__main__":
    main()
