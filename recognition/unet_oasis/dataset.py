"""
Task 2 UNet — OASIS image + segmentation label loading

Directory layout:
    keras_png_slices_{train,validate,test}/       images  case_XXX_slice_Y.nii.png
    keras_png_slices_seg_{train,validate,test}/   labels  seg_XXX_slice_Y.nii.png

The label maps are greyscale PNGs with 4 classes (background + 3 brain tissues); pixel values
are roughly {0, 85, 170, 255}. This module maps them to class indices {0,1,2,3}; training then
converts to one-hot (the task sheet asks for categorical output).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "keras_png_slices_data"
NUM_CLASSES = 4
# Greyscale values that actually occur in the segmentation PNGs (checked against the dataset)
LABEL_VALUES = np.array([0, 85, 170, 255], dtype=np.int16)

# Greyscale value -> class index lookup table. The dataset has only 4 grey levels, and after a
# nearest-neighbour resize those 4 should still be the only values present; a 256-entry LUT does
# the whole mapping in one indexing step, far faster than taking per-pixel distances to the 4
# reference values (which allocates an H*W*4 temporary for every image).
_LABEL_LUT = np.abs(np.arange(256, dtype=np.int16)[:, None] - LABEL_VALUES[None, :]) \
    .argmin(axis=1).astype(np.int64)


class OASISSegmentation(Dataset):
    """Returns (image (1,H,W) float[0,1], mask (H,W) long[0..3])."""

    def __init__(self, root, split: str = "train", image_size: int = 256):
        root = Path(root)
        self.img_dir = root / f"keras_png_slices_{split}"
        self.seg_dir = root / f"keras_png_slices_seg_{split}"
        for d in (self.img_dir, self.seg_dir):
            if not d.exists():
                raise FileNotFoundError(f"directory {d} not found, unzip the dataset first")

        self.img_paths = sorted(self.img_dir.glob("*.png"))
        # pair up by the index in "case_441_slice_0.nii.png" <-> "seg_441_slice_0.nii.png"
        seg_lookup = {p.name.replace("seg_", "", 1): p for p in self.seg_dir.glob("*.png")}
        self.seg_paths = []
        for p in self.img_paths:
            key = p.name.replace("case_", "", 1)
            if key not in seg_lookup:
                raise KeyError(f"{p.name} has no matching label")
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
        seg = self._load(self.seg_paths[idx], nearest=True)   # nearest-neighbour, never interpolate
        # look up the table to map grey values {0,85,170,255} to class indices {0,1,2,3}
        mask = _LABEL_LUT[seg]
        return torch.from_numpy(img)[None], torch.from_numpy(mask)


def to_one_hot(mask: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """(B,H,W) class indices -> (B,C,H,W) one-hot, float."""
    return F.one_hot(mask, num_classes).permute(0, 3, 1, 2).float()


def get_dataloaders(root=DEFAULT_ROOT, batch_size: int = 16, image_size: int = 256,
                    num_workers: int = 4):
    loaders = {}
    for split, shuffle in (("train", True), ("validate", False), ("test", False)):
        ds = OASISSegmentation(root, split, image_size)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=num_workers,
                                    # pin_memory only means anything on CUDA; MPS does not
                                    # support it, so enabling it just prints a warning and
                                    # costs one pointless memory copy
                                    pin_memory=torch.cuda.is_available(),
                                    persistent_workers=num_workers > 0)
        print(f"{split:9s}: {len(ds)} pairs")
    return loaders["train"], loaders["validate"], loaders["test"]


if __name__ == "__main__":
    tr, _, _ = get_dataloaders(num_workers=0, batch_size=4)
    x, y = next(iter(tr))
    print("image:", x.shape, "mask:", y.shape, "classes:", y.unique().tolist())
