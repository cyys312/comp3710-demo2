"""
Task 3 GAN — generator and discriminator (DCGAN architecture plus stabilising changes)

The task sheet states that "GANs have very chaotic convergence" and requires that
"issues such as mode collapse need to be fully resolved". This is therefore not a naive
DCGAN; three targeted stabilisations are added:

1. **R1 gradient penalty** (penalises the squared gradient norm of the discriminator on
   real samples) — lighter than WGAN-GP, as it is computed on real samples only and only
   every k steps. It keeps the discriminator locally Lipschitz, so it cannot grow strong
   enough to make the generator's gradients vanish.
2. **No BatchNorm in the discriminator**; LeakyReLU plus InstanceNorm is used instead, as a
   substitute for spectral normalisation. BatchNorm couples the samples inside one batch,
   a known source of instability in GANs.
3. **EMA of the generator weights** (exponential moving average) — sampling uses the EMA
   weights rather than the current ones. One of the cheapest ways to improve sample quality,
   with almost no downside.
"""

import copy

import torch
import torch.nn as nn


class Generator(nn.Module):
    """z (latent_dim,) -> image (1, image_size, image_size).

    Starts at 4x4; every transposed convolution doubles the side length. image_size=128
    therefore takes 5 upsampling steps.
    """

    def __init__(self, latent_dim: int = 128, base: int = 64, image_size: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        n_up = 0
        size = 4
        while size < image_size:          # how many doublings are needed to reach the target
            size *= 2
            n_up += 1
        if size != image_size:
            raise ValueError(f"image_size must be 4 times a power of 2, got {image_size}")

        # starting width: base * 2^(n_up-1) channels, halved at every layer
        channels = [base * 2 ** i for i in reversed(range(n_up))]

        layers = [
            # a 4x4 transposed conv turns z into a 4x4 map: fewer parameters than Linear+reshape
            nn.ConvTranspose2d(latent_dim, channels[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(True),
        ]
        for in_ch, out_ch in zip(channels, channels[1:]):
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ]
        layers += [
            nn.ConvTranspose2d(channels[-1], 1, 4, 2, 1),
            nn.Tanh(),                    # output in [-1,1], matching the data preprocessing
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    """image (1,H,W) -> one real number with no sigmoid applied (a logit).

    BatchNorm is deliberately avoided: it lets samples in the same batch influence each
    other, a common source of instability in GAN training. InstanceNorm is used instead.
    """

    def __init__(self, base: int = 64, image_size: int = 128):
        super().__init__()
        n_down = 0
        size = image_size
        while size > 4:
            size //= 2
            n_down += 1

        channels = [base * 2 ** i for i in range(n_down)]
        layers = [
            nn.Conv2d(1, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for in_ch, out_ch in zip(channels, channels[1:]):
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [nn.Conv2d(channels[-1], 1, 4, 1, 0)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(module):
    """Initialisation from the DCGAN paper: conv N(0, 0.02), BatchNorm weights N(1, 0.02)."""
    name = module.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def r1_penalty(discriminator, real_images):
    """R1 gradient penalty: squared gradient norm of the discriminator at **real samples**.

    Intuition: flattening the discriminator near the real data manifold stops it from getting
    too sharp; once the discriminator is too strong the generator receives vanishing
    gradients and training collapses immediately.
    Unlike WGAN-GP, R1 needs no interpolation between real and fake samples, so it costs less.
    """
    real_images = real_images.detach().requires_grad_(True)
    logits = discriminator(real_images)
    grad = torch.autograd.grad(
        outputs=logits.sum(), inputs=real_images, create_graph=True)[0]
    return grad.pow(2).flatten(1).sum(1).mean()


class EMA:
    """Exponential moving average of generator weights; sampling from them looks far steadier."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for shadow_p, p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.lerp_(p.detach(), 1.0 - self.decay)
        # buffers (BatchNorm running stats) are copied straight over, not averaged
        for shadow_b, b in zip(self.shadow.buffers(), model.buffers()):
            shadow_b.copy_(b)


@torch.no_grad()
def diversity_score(images: torch.Tensor, max_pairs: int = 2048) -> float:
    """Mean pairwise L2 distance between samples — the diversity score for mode collapse.

    This is the objective evidence this project uses to decide whether mode collapse
    happened: if the generator collapses onto a few modes the samples are nearly identical
    to one another, and this number falls far below the same measure on real data.
    Judging generated images by eye is subjective; this number is evidence you can show.
    """
    flat = images.flatten(1)
    n = flat.size(0)
    idx_a = torch.randint(0, n, (max_pairs,), device=flat.device)
    idx_b = torch.randint(0, n, (max_pairs,), device=flat.device)
    keep = idx_a != idx_b                       # drop pairs of a sample with itself
    return (flat[idx_a[keep]] - flat[idx_b[keep]]).norm(dim=1).mean().item()


if __name__ == "__main__":
    gen, disc = Generator(), Discriminator()
    gen.apply(init_weights)
    z = torch.randn(2, gen.latent_dim)
    fake = gen(z)
    print("generator output:", tuple(fake.shape), "range", (fake.min().item(), fake.max().item()))
    print("discriminator output:", tuple(disc(fake).shape))
    print("generator parameters: %.1f M" % (sum(p.numel() for p in gen.parameters()) / 1e6))
    print("discriminator parameters: %.1f M" % (sum(p.numel() for p in disc.parameters()) / 1e6))
