"""
Task 3 GAN —— OASIS 脑部 MRI 数据加载

与 VAE 的加载器只有一处关键差别：**像素归一化到 [-1, 1] 而不是 [0, 1]**，
因为生成器最后一层用的是 tanh。真假数据的取值范围必须一致，
否则判别器只要看数值范围就能分辨，训练会立刻退化。

GAN 是无监督的，只用原图，不需要 seg_ 标签。
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "keras_png_slices_data"


class OASISImages(Dataset):
    """返回图像张量 (1, H, W)，像素归一化到 [-1, 1]。"""

    def __init__(self, root, split: str = "train", image_size: int = 128):
        self.dir = Path(root) / f"keras_png_slices_{split}"
        if not self.dir.exists():
            raise FileNotFoundError(f"找不到目录 {self.dir}")
        self.paths = sorted(self.dir.glob("*.png"))
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        if self.image_size:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0     # [0,255] -> [-1,1]
        return torch.from_numpy(arr)[None, :, :]


def get_dataloader(root=DEFAULT_ROOT, batch_size: int = 64, image_size: int = 128,
                   num_workers: int = 0, split: str = "train"):
    """GAN 训练只用训练集；没有验证集的概念（没有似然可以早停）。"""
    ds = OASISImages(root, split, image_size)
    print(f"{split}: {len(ds)} images", flush=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


if __name__ == "__main__":
    dl = get_dataloader(batch_size=4)
    x = next(iter(dl))
    print("batch:", tuple(x.shape), "范围", (x.min().item(), x.max().item()))
