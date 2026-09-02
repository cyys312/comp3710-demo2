"""
Task 2 UNet —— OASIS 图像 + 分割标签数据加载

目录结构:
    keras_png_slices_{train,validate,test}/       原图  case_XXX_slice_Y.nii.png
    keras_png_slices_seg_{train,validate,test}/   标签  seg_XXX_slice_Y.nii.png

标签图是灰度 PNG，共 4 类（背景 + 3 种脑组织），像素值约为 {0, 85, 170, 255}。
本模块把它映射成 {0,1,2,3} 的类别索引；训练时再转 one-hot（任务书要求 categorical 输出）。
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "keras_png_slices_data"
NUM_CLASSES = 4
# 分割标签 PNG 中实际出现的灰度值（已在数据集上核对）
LABEL_VALUES = np.array([0, 85, 170, 255], dtype=np.int16)

# 灰度值 -> 类别索引的查找表。数据集只有 4 个灰度档，但最近邻 resize 之后
# 理论上仍只会出现这 4 个值；用 256 项的 LUT 一次索引就能完成映射，
# 比逐像素与 4 个参考值求距离快得多（后者要为每张图分配 H*W*4 的临时数组）。
_LABEL_LUT = np.abs(np.arange(256, dtype=np.int16)[:, None] - LABEL_VALUES[None, :]) \
    .argmin(axis=1).astype(np.int64)


class OASISSegmentation(Dataset):
    """返回 (image (1,H,W) float[0,1], mask (H,W) long[0..3])。"""

    def __init__(self, root, split: str = "train", image_size: int = 256):
        root = Path(root)
        self.img_dir = root / f"keras_png_slices_{split}"
        self.seg_dir = root / f"keras_png_slices_seg_{split}"
        for d in (self.img_dir, self.seg_dir):
            if not d.exists():
                raise FileNotFoundError(f"找不到目录 {d}，请先解压数据集")

        self.img_paths = sorted(self.img_dir.glob("*.png"))
        # 用 "case_441_slice_0.nii.png" <-> "seg_441_slice_0.nii.png" 的编号配对
        seg_lookup = {p.name.replace("seg_", "", 1): p for p in self.seg_dir.glob("*.png")}
        self.seg_paths = []
        for p in self.img_paths:
            key = p.name.replace("case_", "", 1)
            if key not in seg_lookup:
                raise KeyError(f"{p.name} 找不到对应标签")
            self.seg_paths.append(seg_lookup[key])
        self.image_size = image_size

    def __len__(self):
        return len(self.img_paths)

    def _load(self, path, nearest=False):
        img = Image.open(path).convert("L")
        if self.image_size:
            img = img.resize((self.image_size, self.image_size),
                             Image.NEAREST if nearest else Image.BILINEAR)
        return np.asarray(img)

    def __getitem__(self, idx):
        img = self._load(self.img_paths[idx]).astype(np.float32) / 255.0
        seg = self._load(self.seg_paths[idx], nearest=True)   # 标签必须最近邻，不能插值
        # 查表把灰度值 {0,85,170,255} 映射到类别索引 {0,1,2,3}
        mask = _LABEL_LUT[seg]
        return torch.from_numpy(img)[None], torch.from_numpy(mask)


def to_one_hot(mask: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """(B,H,W) 类别索引 -> (B,C,H,W) one-hot，float。"""
    return F.one_hot(mask, num_classes).permute(0, 3, 1, 2).float()


def get_dataloaders(root=DEFAULT_ROOT, batch_size: int = 16, image_size: int = 256,
                    num_workers: int = 4):
    loaders = {}
    for split, shuffle in (("train", True), ("validate", False), ("test", False)):
        ds = OASISSegmentation(root, split, image_size)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=num_workers,
                                    # pin_memory 只对 CUDA 有意义；MPS 不支持，
                                    # 开着只会打印警告并多一次无用的内存拷贝
                                    pin_memory=torch.cuda.is_available(),
                                    persistent_workers=num_workers > 0)
        print(f"{split:9s}: {len(ds)} pairs")
    return loaders["train"], loaders["validate"], loaders["test"]


if __name__ == "__main__":
    tr, _, _ = get_dataloaders(num_workers=0, batch_size=4)
    x, y = next(iter(tr))
    print("image:", x.shape, "mask:", y.shape, "classes:", y.unique().tolist())
