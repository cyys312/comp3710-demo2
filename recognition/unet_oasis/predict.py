"""
Task 2 UNet — test-set inference and visualisation (this is the one run during the demo)

Run:     python recognition/unet_oasis/predict.py --n-show 4
Outputs: outputs/segmentation_examples.png, per-class DSC printed to the terminal
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
    """Prefer CUDA (Rangpur A100), then Apple MPS, falling back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_workers(requested: int, device: torch.device) -> int:
    """Decide how many DataLoader workers to use.

    Measured (M5 Mac, OASIS 256x256): decoding one image takes only ~1 ms, while the IPC
    serialisation overhead of worker processes costs far more than that — num_workers=0
    gives 9 ms/batch, num_workers=6 gives 262 ms/batch, 29x slower; and fork together with
    MPS can leave the main process stuck in an uninterruptible wait.
    Hence: workers only on CUDA (cluster, many CPU cores, data on network storage), 0 elsewhere.
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
    print(f"loaded checkpoint: epoch {state['epoch']}, validation DSC {state.get('dsc')}")
    return model, saved


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(HERE / "checkpoints" / "unet.pth"))
    p.add_argument("--data-root", type=str, default=str(DEFAULT_ROOT))
    p.add_argument("--n-show", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=-1,
                   help="-1 means choose automatically per device (4 on CUDA, 0 on MPS/CPU)")
    args = p.parse_args()

    device = pick_device()
    model, saved = load_model(args.ckpt, device)
    _, _, test_loader = get_dataloaders(args.data_root, 8,
                                        saved.get("image_size", 256), auto_workers(args.num_workers, device))

    # per-class DSC over the whole test set
    inter = torch.zeros(NUM_CLASSES, device=device)
    card = torch.zeros(NUM_CLASSES, device=device)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        i, c = hard_dice_counts(model(x), to_one_hot(y, NUM_CLASSES))
        inter += i
        card += c
    dsc = dice_from_counts(inter, card).cpu()

    print("\n--- Test set Dice similarity coefficient ---")
    for c, v in enumerate(dsc):
        flag = "OK" if v > 0.9 else "below 0.9"
        print(f"  class {c}: {v:.4f}  {flag}")
    print(f"  mean   : {dsc.mean():.4f}")

    # visualise a few examples: input image / ground truth / prediction
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
    print("saved segmentation examples:", out / "segmentation_examples.png")


if __name__ == "__main__":
    main()
