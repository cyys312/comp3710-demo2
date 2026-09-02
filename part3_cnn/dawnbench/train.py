"""
Part 3.2 DAWNBench —— 训练 ResNet-18 / CIFAR-10

目标：测试集准确率 > 90%，训练时间 < 30 分钟（A100 上用混合精度约几分钟即可到 94%）。

关键加速手段：
  * AMP 混合精度 (torch.amp) —— 显著提速且省显存
  * OneCycle 学习率 + SGD(nesterov) —— 少量 epoch 内快速收敛
  * channels_last 内存格式 + cudnn.benchmark

运行:
  python train.py --epochs 30 --batch-size 512
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset import get_dataloaders, get_gpu_loaders
from modules import ResNet18

CKPT_DIR = Path(__file__).parent / "checkpoints"


def pick_device() -> torch.device:
    """优先 CUDA（Rangpur A100），其次 Apple MPS，最后退回 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.4, help="OneCycle 的峰值学习率")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    p.add_argument("--loader", choices=["gpu", "cpu"], default="gpu",
                   help="gpu=整个数据集常驻显存、增强在 GPU 上做（默认，快得多）；"
                        "cpu=torchvision 的 DataLoader + PIL 增强（用于对照）")
    p.add_argument("--log-every", type=int, default=20,
                   help="每多少个 step 打印一次进度")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    args = parse_args()
    device = pick_device()
    torch.backends.cudnn.benchmark = True
    use_amp = (not args.no_amp) and device.type == "cuda"

    # GPU 常驻管线 vs CPU DataLoader —— 见 dataset.GPUCifar 的说明。
    # a100 节点只有 8 个 CPU 核，PIL 增强会成为瓶颈，默认走 GPU 版。
    if args.loader == "gpu":
        train_loader, test_loader, _ = get_gpu_loaders(args.data_root, args.batch_size, device)
    else:
        train_loader, test_loader, _ = get_dataloaders(
            args.data_root, args.batch_size, args.num_workers)

    model = ResNet18().to(device, memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimiser = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.25,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"device={device} amp={use_amp} epochs={args.epochs} batch={args.batch_size} "
          f"loader={args.loader} steps/epoch={len(train_loader)}", flush=True)
    CKPT_DIR.mkdir(exist_ok=True)
    best_acc, t0 = 0.0, time.perf_counter()

    n_steps = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        epoch_start = time.perf_counter()
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            running += loss.item()
            # 第一个 epoch 打分步进度，用来判断数据管线有没有成为瓶颈；
            # 之后只打 epoch 汇总，免得日志太吵。
            if epoch == 1 and (step % args.log_every == 0 or step == n_steps):
                per_step = (time.perf_counter() - epoch_start) / step
                print(f"  epoch 1 [{step:3d}/{n_steps}] loss {running / step:.4f} "
                      f"| {per_step * 1000:.0f} ms/step", flush=True)

        train_time = time.perf_counter() - epoch_start
        acc = evaluate(model, test_loader, device)
        elapsed = time.perf_counter() - t0
        print(f"epoch {epoch:3d} | loss {running / n_steps:.4f} "
              f"| test acc {acc * 100:.2f}% | 本轮 {train_time:.1f}s "
              f"| 累计 {elapsed:.0f}s", flush=True)

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "acc": acc, "epoch": epoch},
                       CKPT_DIR / "resnet18_cifar10.pth")

    total_s = time.perf_counter() - t0
    print(f"\n最佳准确率 {best_acc * 100:.2f}%，总用时 {total_s:.0f} 秒 "
          f"({total_s / 60:.1f} 分钟)")
    # 任务书的三档要求
    print(f"  [1分] >90% 且 <30 分钟          : "
          f"{'达标' if best_acc > 0.90 and total_s < 1800 else '未达标'}")
    print(f"  [2分] >=94% 且 <=360 秒(V100 基准): "
          f"{'达标' if best_acc >= 0.94 and total_s <= 360 else '未达标'}")


if __name__ == "__main__":
    main()
