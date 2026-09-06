"""
Task 3 GAN — OASIS brain MRI data loading

One key difference from the VAE loader: **pixels are normalised to [-1, 1], not [0, 1]**,
because the generator's last layer is a tanh. Real and fake data must span the same range,
otherwise the discriminator can separate them from the value range alone and training
degenerates immediately.

The GAN is unsupervised: only the raw images are used, the seg_ labels are not needed.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "keras_png_slices_data"


class OASISImages(Dataset):
    """Returns image tensors (1, H, W) with pixels normalised to [-1, 1]."""

    def __init__(self, root, split: str = "train", image_size: int = 128):
        self.dir = Path(root) / f"keras_png_slices_{split}"
        if not self.dir.exists():
            raise FileNotFoundError(f"directory not found: {self.dir}")
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
    """Training uses the training split only; no validation set, no likelihood to early-stop on."""
    ds = OASISImages(root, split, image_size)
    print(f"{split}: {len(ds)} images", flush=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


if __name__ == "__main__":
    dl = get_dataloader(batch_size=4)
    x = next(iter(dl))
    print("batch:", tuple(x.shape), "range", (x.min().item(), x.max().item()))
