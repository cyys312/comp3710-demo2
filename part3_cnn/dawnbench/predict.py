"""
Part 3.2 DAWNBench —— 载入 checkpoint 做推理

运行: python predict.py --ckpt checkpoints/resnet18_cifar10.pth
"""

import argparse
from pathlib import Path

import torch

from dataset import get_dataloaders, get_gpu_loaders
from modules import ResNet18



def pick_device() -> torch.device:
    """优先 CUDA（Rangpur A100），其次 Apple MPS，最后退回 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=str(Path(__file__).parent / "checkpoints/resnet18_cifar10.pth"))
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--loader", choices=["gpu", "cpu"], default="gpu")
    args = p.parse_args()

    device = pick_device()
    if args.loader == "gpu":
        _, test_loader, classes = get_gpu_loaders(args.data_root, 512, device)
    else:
        _, test_loader, classes = get_dataloaders(args.data_root, 512, args.num_workers)

    model = ResNet18().to(device, memory_format=torch.channels_last)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"载入 checkpoint: epoch {state['epoch']}, 训练时准确率 {state['acc'] * 100:.2f}%")

    correct = total = 0
    per_class_correct = torch.zeros(len(classes))
    per_class_total = torch.zeros(len(classes))
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for c in range(len(classes)):
                mask = y == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += (pred[mask] == c).sum().item()

    print(f"测试集准确率: {correct / total * 100:.2f}%")
    for c, name in enumerate(classes):
        print(f"  {name:12s} {per_class_correct[c] / per_class_total[c] * 100:5.1f}%")


if __name__ == "__main__":
    main()
