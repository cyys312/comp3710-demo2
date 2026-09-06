"""
Task 2 UNet — model and losses

The point of UNet is the skip connection: the high-resolution features of every encoder
level are concatenated into the matching decoder level, restoring the spatial detail that
downsampling threw away. That is where the segmentation accuracy comes from.

Outputs C=4 channels of logits, used together with one-hot labels (the task sheet asks for
categorical output).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(3x3 conv + BN + ReLU) x 2 — the basic UNet building block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base: int = 32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8]

        self.enc1 = DoubleConv(in_channels, chs[0])
        self.enc2 = DoubleConv(chs[0], chs[1])
        self.enc3 = DoubleConv(chs[1], chs[2])
        self.enc4 = DoubleConv(chs[2], chs[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[3], chs[3] * 2)

        self.up4 = nn.ConvTranspose2d(chs[3] * 2, chs[3], 2, 2)
        self.dec4 = DoubleConv(chs[3] * 2, chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, 2)
        self.dec3 = DoubleConv(chs[2] * 2, chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, 2)
        self.dec2 = DoubleConv(chs[1] * 2, chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, 2)
        self.dec1 = DoubleConv(chs[0] * 2, chs[0])

        self.head = nn.Conv2d(chs[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))   # skip connection
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)                                   # logits (B, C, H, W)


def dice_per_class(logits, target_one_hot, eps: float = 1e-6):
    """**Soft** Dice: computed straight from the softmax probabilities, differentiable, and
    meant only for the loss.

    Do not report it as an evaluation metric — the DSC the marking asks for is computed on
    **hard predictions** (the argmax class map). Soft Dice reads too low when the model is
    unconfident and too high when it is overconfident.
    """
    probs = F.softmax(logits, dim=1)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_one_hot, dims)
    cardinality = torch.sum(probs + target_one_hot, dims)
    return (2.0 * intersection + eps) / (cardinality + eps)


@torch.no_grad()
def hard_dice_counts(logits, target_one_hot):
    """Return per-class (intersection, cardinality) counts, to be accumulated across batches.

    Why counts rather than the DSC itself: DSC is a ratio, so "compute the DSC per batch and
    average" is not the same as "the DSC over the whole dataset", and the short final batch
    would carry the same weight as a full one. The correct approach is to accumulate the
    numerator and denominator separately and divide once at the end.
    """
    num_classes = logits.shape[1]
    pred = F.one_hot(logits.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(pred * target_one_hot, dims)
    cardinality = torch.sum(pred + target_one_hot, dims)
    return intersection, cardinality


def dice_from_counts(intersection, cardinality, eps: float = 1e-6):
    """Turn the accumulated counts into per-class DSC."""
    return (2.0 * intersection + eps) / (cardinality + eps)


def dice_loss(logits, target_one_hot):
    """1 - mean DSC. Optimises the metric directly; suits class imbalance better than CE alone."""
    return 1.0 - dice_per_class(logits, target_one_hot).mean()


class CombinedLoss(nn.Module):
    """Cross-entropy + Dice, covering both per-pixel correctness and region overlap."""

    def __init__(self, ce_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def forward(self, logits, target_idx, target_one_hot):
        return self.ce_weight * self.ce(logits, target_idx) + \
               (1 - self.ce_weight) * dice_loss(logits, target_one_hot)
