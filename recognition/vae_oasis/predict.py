"""
Task 1 VAE — inference and manifold visualisation

Three visualisations, as the marking criteria ask to "visualise the resulting manifold":
  1. reconstruction — originals vs reconstructions, a check that the model learnt something
  2. grid          — when latent_dim == 2, decode a [-3,3]^2 grid to see the manifold directly
  3. umap          — high-dimensional latent space reduced to 2D by UMAP, then scattered
                     (needs pip install umap-learn)

Run:
  python recognition/vae_oasis/predict.py --mode umap
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image

from dataset import DEFAULT_ROOT, get_dataloaders
from modules import VAE

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


def normal_ppf(q):
    """Quantile function (probit) of the standard normal distribution.

    Equivalent to scipy.stats.norm.ppf but written with numpy alone — the cluster's conda
    environment has no scipy, and dragging in a whole dependency for one function is not worth it.
    ppf(q) = sqrt(2) * erfinv(2q - 1); numpy has no erfinv, so this borrows torch's
    (a few dozen points on CPU, negligible cost).
    """
    q = torch.as_tensor(np.asarray(q, dtype=np.float64))
    return (torch.sqrt(torch.tensor(2.0, dtype=torch.float64)) *
            torch.erfinv(2 * q - 1)).numpy()


def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    saved = state.get("args", {})
    model = VAE(saved.get("latent_dim", 32), image_size=saved.get("image_size", 128)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, saved


@torch.no_grad()
def show_reconstructions(model, loader, device, out):
    x = next(iter(loader))[:8].to(device)
    recon, _, _ = model(x)
    save_image(torch.cat([x, recon]), out, nrow=8)
    print("saved reconstruction comparison:", out)


@torch.no_grad()
def manifold_grid(model, device, out, n: int = 20):
    """When latent_dim == 2, tile a grid of standard-normal quantiles and decode it."""
    if model.latent_dim != 2:
        print(f"latent_dim={model.latent_dim}: the grid needs 2D, use --mode umap instead")
        return
    # Space the grid by quantile rather than uniformly: the prior is standard normal, so equal
    # quantile steps cover the probability mass evenly and the edges are not untrained territory.
    grid_x = normal_ppf(np.linspace(0.02, 0.98, n))
    grid_y = normal_ppf(np.linspace(0.02, 0.98, n))
    zs = torch.tensor([[xi, yi] for yi in grid_y for xi in grid_x],
                      dtype=torch.float32, device=device)
    imgs = model.decoder(zs).cpu()
    save_image(imgs, out, nrow=n)
    print("saved 2D manifold grid:", out)


@torch.no_grad()
def encode_dataset(model, loader, device, max_batches: int = 20):
    """Encode a few batches into a latent matrix (N, latent_dim), taking the posterior mean mu."""
    latents = []
    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        mu, _ = model.encoder(x.to(device))
        latents.append(mu.cpu())
    return torch.cat(latents)


def pca_2d(latents: torch.Tensor):
    """Reduce the latent vectors to 2D (PCA); returns (projection, variance ratio of each PC).

    A fallback for UMAP: the cluster's conda environment has no umap-learn, whereas PCA needs
    only torch's SVD and pulls in no extra dependency.
    For the question of whether the latent space has continuous structure, a linear projection
    already settles it; UMAP's advantage is preserving non-linear neighbourhood structure, which
    is nicer to have but does not change the conclusion.
    """
    centred = latents - latents.mean(dim=0, keepdim=True)
    # full_matrices=False computes only the singular vectors we need
    u, s, _ = torch.linalg.svd(centred, full_matrices=False)
    projected = u[:, :2] * s[:2]
    ratio = (s ** 2 / (s ** 2).sum())[:2]
    return projected.numpy(), ratio.numpy()


@torch.no_grad()
def manifold_umap(model, loader, device, out, max_batches: int = 20):
    """High-dimensional latent space -> 2D scatter. Prefers UMAP, falls back to PCA if absent."""
    latents = encode_dataset(model, loader, device, max_batches)

    try:
        import umap
        embedding = umap.UMAP(n_neighbors=15, min_dist=0.1,
                              random_state=42).fit_transform(latents.numpy())
        method, subtitle = "UMAP", ""
    except ImportError:
        embedding, ratio = pca_2d(latents)
        method = "PCA"
        subtitle = (f"\nPC1 {ratio[0] * 100:.1f}% + PC2 {ratio[1] * 100:.1f}% "
                    f"= {ratio.sum() * 100:.1f}% of latent variance")
        print("umap-learn not installed, falling back to a PCA projection (conclusion unchanged)")

    plt.figure(figsize=(7, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=4, alpha=0.5)
    plt.title(f"{method} projection of VAE latent space "
              f"(n={len(latents)}, latent_dim={model.latent_dim}){subtitle}")
    plt.xlabel(f"{method} 1")
    plt.ylabel(f"{method} 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"saved {method} latent space projection:", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(HERE / "checkpoints" / "vae_z32.pth"),
                   help="for the latent_dim=2 model use checkpoints/vae_z2.pth")
    p.add_argument("--data-root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--mode", choices=["recon", "grid", "umap", "all"], default="all")
    p.add_argument("--num-workers", type=int, default=-1,
                   help="-1 picks automatically per device (4 on CUDA, 0 on MPS/CPU)")
    args = p.parse_args()

    device = pick_device()
    model, saved = load_model(args.ckpt, device)
    _, _, test_loader = get_dataloaders(args.data_root, 64,
                                        saved.get("image_size", 128), auto_workers(args.num_workers, device))
    outdir = HERE / "outputs"
    outdir.mkdir(exist_ok=True)

    tag = f"z{model.latent_dim}"
    if args.mode in ("recon", "all"):
        show_reconstructions(model, test_loader, device, outdir / f"{tag}_reconstruction.png")
    if args.mode in ("grid", "all"):
        manifold_grid(model, device, outdir / f"{tag}_manifold_grid.png")
    if args.mode in ("umap", "all"):
        manifold_umap(model, test_loader, device, outdir / f"{tag}_manifold_umap.png")


if __name__ == "__main__":
    main()
