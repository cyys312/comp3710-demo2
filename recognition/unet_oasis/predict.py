"""
Task 2 UNet —— 测试集推理与可视化（demo 现场要跑这个）

运行: python recognition/unet_oasis/predict.py --n-show 4
产物: outputs/segmentation_examples.png、逐类 DSC 打印
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import DEFAULT_ROOT, NUM_CLASSES, get_dataloaders, to_one_hot
from modules import UNet, dice_from_counts, hard_dice_counts

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


def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    saved = state.get("args", {})
    model = UNet(1, NUM_CLASSES, saved.get("base", 32)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"载入 checkpoint: epoch {state['epoch']}, 验证 DSC {state.get('dsc')}")
    return model, saved


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(HERE / "checkpoints" / "unet.pth"))
    p.add_argument("--data-root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--n-show", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=-1,
                   help="-1 表示按设备自动选择（CUDA 用 4，MPS/CPU 用 0）")
    args = p.parse_args()

    device = pick_device()
    model, saved = load_model(args.ckpt, device)
    _, _, test_loader = get_dataloaders(args.data_root, 8,
                                        saved.get("image_size", 256), auto_workers(args.num_workers, device))

    # 整个测试集的逐类 DSC
    inter = torch.zeros(NUM_CLASSES, device=device)
    card = torch.zeros(NUM_CLASSES, device=device)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        i, c = hard_dice_counts(model(x), to_one_hot(y, NUM_CLASSES))
        inter += i
        card += c
    dsc = dice_from_counts(inter, card).cpu()

    print("\n--- 测试集 Dice 相似系数 ---")
    for c, v in enumerate(dsc):
        flag = "OK" if v > 0.9 else "低于 0.9"
        print(f"  class {c}: {v:.4f}  {flag}")
    print(f"  mean   : {dsc.mean():.4f}")

    # 可视化若干样例：原图 / 真值 / 预测
    x, y = next(iter(test_loader))
    pred = model(x.to(device)).argmax(1).cpu()
    n = min(args.n_show, x.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    axes = axes.reshape(n, 3)
    for i in range(n):
        axes[i, 0].imshow(x[i, 0], cmap="gray"); axes[i, 0].set_title("MRI")
        axes[i, 1].imshow(y[i], vmin=0, vmax=NUM_CLASSES - 1); axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(pred[i], vmin=0, vmax=NUM_CLASSES - 1); axes[i, 2].set_title("Prediction")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    out = HERE / "outputs"; out.mkdir(exist_ok=True)
    fig.savefig(out / "segmentation_examples.png", dpi=120)
    print("已保存分割样例:", out / "segmentation_examples.png")


if __name__ == "__main__":
    main()
