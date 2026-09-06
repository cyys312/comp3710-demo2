"""
Task 1 VAE — OASIS brain MRI data loading

Data directory (after extracting locally):
    data/keras_png_slices_data/
        keras_png_slices_train/      raw slices case_XXX_slice_Y.nii.png
        keras_png_slices_validate/
        keras_png_slices_test/
On the Rangpur cluster this lives at /home/groups/comp3710/keras_png_slices_data/

The VAE is unsupervised: it uses the raw slices only and needs no seg_ labels.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Default local path: relative to the repository root
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "keras_png_slices_data"


class OASISImages(Dataset):
    """Returns the image tensor (1, H, W) only, with pixels normalised to [0, 1]."""

    def __init__(self, root, split: str = "train", image_size: int = 128):
        self.dir = Path(root) / f"keras_png_slices_{split}"
        if not self.dir.exists():
            raise FileNotFoundError(f"directory {self.dir} not found, extract the dataset first")
        self.paths = sorted(self.dir.glob("*.png"))
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        if self.image_size:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None, :, :]


def get_dataloaders(root=DEFAULT_ROOT, batch_size: int = 64, image_size: int = 128,
                    num_workers: int = 4):
    loaders = {}
    for split, shuffle in (("train", True), ("validate", False), ("test", False)):
        ds = OASISImages(root, split, image_size)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                    num_workers=num_workers,
                                    pin_memory=torch.cuda.is_available(),
                                    persistent_workers=num_workers > 0)
        print(f"{split:9s}: {len(ds)} images")
    return loaders["train"], loaders["validate"], loaders["test"]


if __name__ == "__main__":
    tr, va, te = get_dataloaders(num_workers=0)
    x = next(iter(tr))
    print("batch:", x.shape, x.min().item(), x.max().item())
