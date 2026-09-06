"""
Part 3.2 DAWNBench — CIFAR-10 data loading and augmentation

The augmentation follows the usual DAWNBench recipe: random crop (padding=4) + horizontal
flip + normalisation.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Per-channel mean/std of the CIFAR-10 training set
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
    """Return (train_loader, test_loader, classes).

    If cifar-10-batches-py already exists under root it is read directly, with no network
    access. On the cluster we point at /home/groups/cifar/CIFAR-10 (a read-only shared copy),
    so download=True must not be passed — torchvision would try to write into that read-only
    directory and fail.
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
# GPU-resident data pipeline
# =====================================================================
class GPUCifar:
    """Keep all of CIFAR-10 in GPU memory; crop/flip/normalise entirely with tensor ops.

    Why this is needed: each a100 node on Rangpur has only 8 CPU cores, and torchvision's PIL
    augmentation pipeline (RandomCrop -> Flip -> ToTensor -> Normalize) sustains only about
    1-2k img/s with 4 workers. An epoch then spends tens of seconds just waiting for data
    while the A100 sits idle. DAWNBench demands 94% within the V100's 360 seconds, so this
    bottleneck has to go.

    Why it is feasible: the raw CIFAR-10 pixels total only 50000 x 3 x 32 x 32 = 153 MB
    (uint8), negligible against 40 GB of GPU memory. Once moved across in one shot, per-batch
    augmentation is a handful of GPU ops with negligible cost, and the CPU leaves the hot path
    entirely.

    Equivalence with the CPU version:
      * Random crop — reflect-pad to 40x40, then draw an independent random offset in [0,8]
        per sample and crop back to 32x32; per-sample equivalent to T.RandomCrop(32,
        padding=4, padding_mode="reflect") (an offset of 4 reproduces the original exactly).
      * Horizontal flip — each sample flipped independently with probability 0.5, as in
        T.RandomHorizontalFlip().
      * Normalisation — the same MEAN/STD, only scaled back up by 255 so they apply directly
        to the float pixels converted from uint8 (torchvision's ToTensor divides by 255 first).
    """

    def __init__(self, root: str, train: bool, batch_size: int, device,
                 augment: bool | None = None):
        need_download = not (Path(root) / "cifar-10-batches-py").is_dir()
        ds = torchvision.datasets.CIFAR10(root, train=train, download=need_download)

        # (N,32,32,3) uint8 -> (N,3,32,32) uint8, moved into GPU memory in one go
        images = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
        self.images = images.to(device, non_blocking=True)
        self.targets = torch.tensor(ds.targets, dtype=torch.long, device=device)
        self.classes = ds.classes

        self.device = device
        self.batch_size = batch_size
        self.augment = train if augment is None else augment
        # ToTensor divides pixels by 255; we skip that and scale MEAN/STD up by 255 instead
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255.0
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255.0

    def __len__(self):
        """Batches per epoch. Training drops the short final batch, matching drop_last=True."""
        if self.augment:
            return len(self.images) // self.batch_size
        return (len(self.images) + self.batch_size - 1) // self.batch_size

    def _random_crop(self, x):
        """Per-sample reflect-padded random crop, equivalent to T.RandomCrop(32, padding=4)."""
        n = x.shape[0]
        dev = x.device
        padded = F.pad(x, (4, 4, 4, 4), mode="reflect")          # (N,3,40,40)
        offset_y = torch.randint(0, 9, (n,), device=dev)
        offset_x = torch.randint(0, 9, (n,), device=dev)
        rows = offset_y.view(n, 1, 1) + torch.arange(32, device=dev).view(1, 32, 1)
        cols = offset_x.view(n, 1, 1) + torch.arange(32, device=dev).view(1, 1, 32)
        batch_idx = torch.arange(n, device=dev).view(n, 1, 1)
        # Mixing advanced indexing with a slice moves the advanced dims to the front -> (N,32,32,3)
        cropped = padded[batch_idx, :, rows, cols]
        return cropped.permute(0, 3, 1, 2).contiguous()

    def _random_flip(self, x):
        """Per-sample horizontal flip at probability 0.5, same as T.RandomHorizontalFlip()."""
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
    """GPU-resident version of (train_loader, test_loader, classes)."""
    train = GPUCifar(root, train=True, batch_size=batch_size, device=device)
    test = GPUCifar(root, train=False, batch_size=batch_size, device=device, augment=False)
    return train, test, train.classes
