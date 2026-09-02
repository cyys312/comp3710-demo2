"""
Part 3.2 DAWNBench —— CIFAR-10 数据加载与增强

增强策略沿用 DAWNBench 常用配置：随机裁剪（padding=4）+ 水平翻转 + 标准化。
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# CIFAR-10 训练集的逐通道均值/标准差
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_transforms():
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    return train_tf, test_tf


def get_dataloaders(root: str = "./data", batch_size: int = 512, num_workers: int = 8):
    """返回 (train_loader, test_loader, classes)。

    root 下若已存在 cifar-10-batches-py 就直接读，不联网。集群上用
    /home/groups/cifar/CIFAR-10（只读的共享副本），因此不能传 download=True ——
    torchvision 会尝试往只读目录写而报错。
    """
    train_tf, test_tf = get_transforms()
    need_download = not (Path(root) / "cifar-10-batches-py").is_dir()
    train_set = torchvision.datasets.CIFAR10(root, train=True, download=need_download,
                                             transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root, train=False, download=need_download,
                                            transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader, train_set.classes


# =====================================================================
# GPU 常驻数据管线
# =====================================================================
class GPUCifar:
    """把整个 CIFAR-10 放进显存，随机裁剪/翻转/归一化全部用张量算子完成。

    为什么需要它：Rangpur 的 a100 节点每个只有 8 个 CPU 核，
    torchvision 那条 PIL 增强管线（RandomCrop -> Flip -> ToTensor -> Normalize）
    在 4 个 worker 下只有大约 1~2k img/s，一个 epoch 光等数据就要几十秒，
    A100 大部分时间在空转。DAWNBench 要求 94% 且时间不超过 V100 的 360 秒，
    这个瓶颈必须消掉。

    可行性：CIFAR-10 的原始像素一共只有 50000 x 3 x 32 x 32 = 153 MB（uint8），
    相对 40 GB 显存微不足道。一次性搬进去之后，每个 batch 的增强都是
    几个 GPU 算子，开销可以忽略，CPU 彻底退出热路径。

    与 CPU 版的等价性：
      * 随机裁剪 —— 先 reflect padding 到 40x40，再对每个样本独立取
        [0,8] 的随机偏移裁回 32x32，和 T.RandomCrop(32, padding=4,
        padding_mode="reflect") 逐样本等价（偏移=4 时精确还原原图）。
      * 水平翻转 —— 每个样本独立以 0.5 的概率翻转，同 T.RandomHorizontalFlip()。
      * 归一化 —— 同一组 MEAN/STD，只是先乘回 255 以便直接作用在 uint8 转来的
        浮点像素上（torchvision 的 ToTensor 会先除 255）。
    """

    def __init__(self, root: str, train: bool, batch_size: int, device,
                 augment: bool | None = None):
        need_download = not (Path(root) / "cifar-10-batches-py").is_dir()
        ds = torchvision.datasets.CIFAR10(root, train=train, download=need_download)

        # (N,32,32,3) uint8 -> (N,3,32,32) uint8，一次性搬进显存
        images = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
        self.images = images.to(device, non_blocking=True)
        self.targets = torch.tensor(ds.targets, dtype=torch.long, device=device)
        self.classes = ds.classes

        self.device = device
        self.batch_size = batch_size
        self.augment = train if augment is None else augment
        # ToTensor 会把像素除以 255，这里省掉这一步，直接把 MEAN/STD 乘回 255
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255.0
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255.0

    def __len__(self):
        """一个 epoch 的 batch 数。训练时丢掉不满的尾批，与 drop_last=True 一致。"""
        if self.augment:
            return len(self.images) // self.batch_size
        return (len(self.images) + self.batch_size - 1) // self.batch_size

    def _random_crop(self, x):
        """逐样本的 reflect-padding 随机裁剪，等价于 T.RandomCrop(32, padding=4)。"""
        n = x.shape[0]
        dev = x.device
        padded = F.pad(x, (4, 4, 4, 4), mode="reflect")          # (N,3,40,40)
        offset_y = torch.randint(0, 9, (n,), device=dev)
        offset_x = torch.randint(0, 9, (n,), device=dev)
        rows = offset_y.view(n, 1, 1) + torch.arange(32, device=dev).view(1, 32, 1)
        cols = offset_x.view(n, 1, 1) + torch.arange(32, device=dev).view(1, 1, 32)
        batch_idx = torch.arange(n, device=dev).view(n, 1, 1)
        # 混用高级索引与切片时，高级索引的维度会排到最前面 -> (N,32,32,3)
        cropped = padded[batch_idx, :, rows, cols]
        return cropped.permute(0, 3, 1, 2).contiguous()

    def _random_flip(self, x):
        """逐样本 0.5 概率水平翻转，等价于 T.RandomHorizontalFlip()。"""
        do_flip = torch.rand(x.shape[0], device=x.device) < 0.5
        return torch.where(do_flip.view(-1, 1, 1, 1), x.flip(-1), x)

    def __iter__(self):
        n = len(self.images)
        if self.augment:
            order = torch.randperm(n, device=self.device)[: len(self) * self.batch_size]
        else:
            order = torch.arange(n, device=self.device)

        for start in range(0, len(order), self.batch_size):
            idx = order[start: start + self.batch_size]
            x = self.images[idx].float()
            if self.augment:
                x = self._random_flip(self._random_crop(x))
            x = (x - self.mean) / self.std
            yield x.contiguous(memory_format=torch.channels_last), self.targets[idx]


def get_gpu_loaders(root: str, batch_size: int, device):
    """GPU 常驻版的 (train_loader, test_loader, classes)。"""
    train = GPUCifar(root, train=True, batch_size=batch_size, device=device)
    test = GPUCifar(root, train=False, batch_size=batch_size, device=device, augment=False)
    return train, test, train.classes
