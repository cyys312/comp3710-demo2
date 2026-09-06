"""
Task 1 VAE — model definitions

VAE = encoder q(z|x) -> reparameterised sample z -> decoder p(x|z)
Loss = reconstruction error + KL divergence (pulls the posterior towards the standard normal prior)

TODO: to visualise the manifold directly on a 2D grid, set latent_dim to 2;
      higher dimensions (16/32) reconstruct more sharply but need UMAP before plotting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Four stride=2 convolutions squeeze 128x128 down to 8x8, then emit mu and logvar."""

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
    """Transposed-convolution upsampling, mirroring the encoder."""

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
        return torch.sigmoid(self.net(h))      # map the output back to [0,1]


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 32, base: int = 32, image_size: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, base, image_size)
        self.decoder = Decoder(latent_dim, base, image_size)

    @staticmethod
    def reparameterise(mu, logvar):
        """Reparameterisation trick: z = mu + sigma * eps, which keeps sampling differentiable."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta: float = 1.0):
    """Negative ELBO: reconstruction term (BCE) + beta * KL term. Returns (total, recon, KL)."""
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl, recon_loss, kl
