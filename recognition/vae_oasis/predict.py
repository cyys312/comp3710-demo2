"""
Task 1 VAE —— 推理与流形 (manifold) 可视化

三种可视化，评分要求「visualise the resulting manifold」：
  1. reconstruction —— 原图 vs 重建，检查模型是否学到东西
  2. grid          —— latent_dim == 2 时，在 [-3,3]^2 网格上解码，直接看流形
  3. umap          —— 高维隐空间先用 UMAP 降到 2D 再散点（需 pip install umap-learn）

运行:
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


def normal_ppf(q):
    """标准正态分布的分位数函数（probit）。

    等价于 scipy.stats.norm.ppf，但只用 numpy 实现 —— 集群的 conda 环境里
    没装 scipy，为一个函数拉一整个依赖不值得。
    ppf(q) = sqrt(2) * erfinv(2q - 1)，erfinv 用 numpy 没有，
    这里借 torch 的 erfinv（CPU 上算几十个点，开销可以忽略）。
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
    print("已保存重建对比:", out)


@torch.no_grad()
def manifold_grid(model, device, out, n: int = 20):
    """latent_dim == 2 时，按标准正态分位数铺网格并解码。"""
    if model.latent_dim != 2:
        print(f"latent_dim={model.latent_dim}，网格法只适用于 2 维，改用 --mode umap")
        return
    # 按分位数（而不是等距）铺网格：先验是标准正态，等分位数采样
    # 才能让网格点均匀覆盖概率质量，边缘不会全是没训练过的区域。
    grid_x = normal_ppf(np.linspace(0.02, 0.98, n))
    grid_y = normal_ppf(np.linspace(0.02, 0.98, n))
    zs = torch.tensor([[xi, yi] for yi in grid_y for xi in grid_x],
                      dtype=torch.float32, device=device)
    imgs = model.decoder(zs).cpu()
    save_image(imgs, out, nrow=n)
    print("已保存 2D 流形网格:", out)


@torch.no_grad()
def encode_dataset(model, loader, device, max_batches: int = 20):
    """把若干个 batch 编码成隐向量矩阵 (N, latent_dim)，取后验均值 mu。"""
    latents = []
    for i, x in enumerate(loader):
        if i >= max_batches:
            break
        mu, _ = model.encoder(x.to(device))
        latents.append(mu.cpu())
    return torch.cat(latents)


def pca_2d(latents: torch.Tensor):
    """把隐向量降到 2 维（PCA），返回 (投影, 两个主成分各自解释的方差比例）。

    作为 UMAP 的兜底：集群的 conda 环境里没有 umap-learn，
    而 PCA 用 torch 的 SVD 就能做，不引入任何额外依赖。
    对「隐空间有没有连续结构」这个问题，线性投影已经足够说明问题；
    UMAP 的优势在于保留非线性的邻域结构，有它更好，没有也不影响结论。
    """
    centred = latents - latents.mean(dim=0, keepdim=True)
    # full_matrices=False 只算需要的奇异向量
    u, s, _ = torch.linalg.svd(centred, full_matrices=False)
    projected = u[:, :2] * s[:2]
    ratio = (s ** 2 / (s ** 2).sum())[:2]
    return projected.numpy(), ratio.numpy()


@torch.no_grad()
def manifold_umap(model, loader, device, out, max_batches: int = 20):
    """高维隐空间 -> 2D 散点。优先 UMAP，没装就退回 PCA。"""
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
        print("未装 umap-learn，改用 PCA 投影（结论不受影响）")

    plt.figure(figsize=(7, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=4, alpha=0.5)
    plt.title(f"{method} projection of VAE latent space "
              f"(n={len(latents)}, latent_dim={model.latent_dim}){subtitle}")
    plt.xlabel(f"{method} 1")
    plt.ylabel(f"{method} 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"已保存 {method} 隐空间投影:", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(HERE / "checkpoints" / "vae_z32.pth"),
                   help="latent_dim=2 的模型用 checkpoints/vae_z2.pth")
    p.add_argument("--data-root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--mode", choices=["recon", "grid", "umap", "all"], default="all")
    p.add_argument("--num-workers", type=int, default=-1,
                   help="-1 表示按设备自动选择（CUDA 用 4，MPS/CPU 用 0）")
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
