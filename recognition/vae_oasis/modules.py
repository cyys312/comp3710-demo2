"""
Task 1 VAE —— 模型定义

VAE = 编码器 q(z|x) -> 重参数化采样 z -> 解码器 p(x|z)
损失 = 重建误差 + KL 散度（把后验拉向标准正态先验）

TODO: 若要用 2D 网格直接可视化 manifold（流形），把 latent_dim 设为 2；
      更高维（如 16/32）重建更清晰，但需要 UMAP 降维后再画。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """4 次 stride=2 卷积把 128x128 压到 8x8，再输出 mu 与 logvar。"""

    def __init__(self, latent_dim: int = 32, base: int = 32, image_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU(True),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU(True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.BatchNorm2d(base * 4), nn.ReLU(True),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1), nn.BatchNorm2d(base * 8), nn.ReLU(True),
        )
        self.feat_size = image_size // 16
        flat = base * 8 * self.feat_size ** 2
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """与编码器对称的转置卷积上采样。"""

    def __init__(self, latent_dim: int = 32, base: int = 32, image_size: int = 128):
        super().__init__()
        self.base, self.feat_size = base, image_size // 16
        self.fc = nn.Linear(latent_dim, base * 8 * self.feat_size ** 2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1), nn.BatchNorm2d(base * 4), nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU(True),
            nn.ConvTranspose2d(base, 1, 4, 2, 1),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, self.base * 8, self.feat_size, self.feat_size)
        return torch.sigmoid(self.net(h))      # 输出映射回 [0,1]


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 32, base: int = 32, image_size: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, base, image_size)
        self.decoder = Decoder(latent_dim, base, image_size)

    @staticmethod
    def reparameterise(mu, logvar):
        """重参数化技巧：z = mu + sigma * eps，让采样这一步可导。"""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta: float = 1.0):
    """ELBO 的负值：重建项 (BCE) + beta * KL 项。返回 (总损失, 重建, KL)。"""
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl, recon_loss, kl
