"""
Task 2 UNet —— 模型与损失

UNet 的关键是 skip connection（跳跃连接）：把编码器每层的高分辨率特征
拼接到对应的解码器层，弥补下采样丢掉的空间细节，这是分割精度的来源。

输出为 C=4 通道的 logits，配合 one-hot 标签使用（任务书要求 categorical 输出）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(3x3 卷积 + BN + ReLU) x 2 —— UNet 的基本单元。"""

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
    """**软** Dice：直接用 softmax 概率算，可导，专门给损失函数用。

    注意不要拿它当评测指标上报 —— 评分要求的 DSC 是对**硬预测**（argmax 之后
    的类别图）算的。软 Dice 在模型不自信时会偏低，在过分自信时又会偏高。
    """
    probs = F.softmax(logits, dim=1)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_one_hot, dims)
    cardinality = torch.sum(probs + target_one_hot, dims)
    return (2.0 * intersection + eps) / (cardinality + eps)


@torch.no_grad()
def hard_dice_counts(logits, target_one_hot):
    """返回逐类的 (交集, 基数) 计数，用于跨 batch 累加。

    为什么要返回计数而不是直接返回 DSC：DSC 是个比值，
    「先按 batch 算 DSC 再取平均」并不等于「整个数据集上的 DSC」，
    而且最后一个不满的 batch 会被赋予同等权重。正确做法是把分子分母
    分别累加完，最后再做一次除法。
    """
    num_classes = logits.shape[1]
    pred = F.one_hot(logits.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(pred * target_one_hot, dims)
    cardinality = torch.sum(pred + target_one_hot, dims)
    return intersection, cardinality


def dice_from_counts(intersection, cardinality, eps: float = 1e-6):
    """把累加好的计数换算成逐类 DSC。"""
    return (2.0 * intersection + eps) / (cardinality + eps)


def dice_loss(logits, target_one_hot):
    """1 - 平均 DSC。直接优化评测指标，比纯交叉熵更适合类别不平衡的分割。"""
    return 1.0 - dice_per_class(logits, target_one_hot).mean()


class CombinedLoss(nn.Module):
    """交叉熵 + Dice，兼顾像素级正确率与区域重叠度。"""

    def __init__(self, ce_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def forward(self, logits, target_idx, target_one_hot):
        return self.ce_weight * self.ce(logits, target_idx) + \
               (1 - self.ce_weight) * dice_loss(logits, target_one_hot)
