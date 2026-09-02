"""
Task 3 GAN —— 生成器与判别器（DCGAN 架构 + 稳定化改动）

任务书对这一项的原话是「GANs have very chaotic convergence」，且要求
「issues such as mode collapse need to be fully resolved」。因此这里不是
朴素 DCGAN，而是加了三处针对性的稳定化：

1. **R1 梯度惩罚**（在真实样本上惩罚判别器梯度的平方范数）
   —— 比 WGAN-GP 更轻量，只在真实样本上算，且每 k 步做一次即可。
   作用是把判别器约束成局部 Lipschitz，避免它变得过强导致生成器梯度消失。
2. **判别器不用 BatchNorm**，改用 LeakyReLU + 谱归一化的替代方案 InstanceNorm。
   BatchNorm 会让判别器对同一个 batch 内的样本产生耦合，
   在 GAN 里是已知的不稳定来源。
3. **生成器权重的 EMA**（指数滑动平均）—— 采样时用 EMA 权重而非当前权重。
   这是提升生成质量最省事的手段之一，几乎没有副作用。
"""

import copy

import torch
import torch.nn as nn


class Generator(nn.Module):
    """z (latent_dim,) -> 图像 (1, image_size, image_size)。

    从 4x4 开始，每层转置卷积把边长翻倍。image_size=128 时共 5 次上采样。
    """

    def __init__(self, latent_dim: int = 128, base: int = 64, image_size: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        n_up = 0
        size = 4
        while size < image_size:          # 需要几次翻倍才能到目标尺寸
            size *= 2
            n_up += 1
        if size != image_size:
            raise ValueError(f"image_size 必须是 4 的 2 次幂倍，收到 {image_size}")

        # 起始通道数：base * 2^(n_up-1)，逐层减半
        channels = [base * 2 ** i for i in reversed(range(n_up))]

        layers = [
            # 用 4x4 转置卷积把 z 展开成 4x4 特征图，比先 Linear 再 reshape 更省参数
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
            nn.Tanh(),                    # 输出到 [-1,1]，与数据预处理保持一致
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    """图像 (1,H,W) -> 一个未过 sigmoid 的实数（logit）。

    刻意不用 BatchNorm：它会让同一 batch 内的样本互相影响，
    在 GAN 训练中是常见的不稳定来源。这里用 InstanceNorm 代替。
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
    """DCGAN 论文的初始化：卷积 N(0, 0.02)，BatchNorm 权重 N(1, 0.02)。"""
    name = module.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def r1_penalty(discriminator, real_images):
    """R1 梯度惩罚：判别器在**真实样本**处梯度的平方范数。

    直觉：在真实数据流形附近把判别器压平，它就不会变得过于尖锐；
    判别器一旦过强，生成器收到的梯度会消失，训练随即崩掉。
    相比 WGAN-GP，R1 不需要在真假样本之间插值，开销更小。
    """
    real_images = real_images.detach().requires_grad_(True)
    logits = discriminator(real_images)
    grad = torch.autograd.grad(
        outputs=logits.sum(), inputs=real_images, create_graph=True)[0]
    return grad.pow(2).flatten(1).sum(1).mean()


class EMA:
    """生成器权重的指数滑动平均。采样时用 EMA 权重，画面明显更稳。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for shadow_p, p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.lerp_(p.detach(), 1.0 - self.decay)
        # buffer（BatchNorm 的 running stats）直接拷贝，不做平均
        for shadow_b, b in zip(self.shadow.buffers(), model.buffers()):
            shadow_b.copy_(b)


@torch.no_grad()
def diversity_score(images: torch.Tensor, max_pairs: int = 2048) -> float:
    """样本两两之间的平均 L2 距离 —— 用来量化 mode collapse。

    这是本项目判断「有没有模式崩塌」的客观依据：若生成器崩塌到少数几个模式，
    生成样本彼此几乎相同，这个数会远低于真实数据的同一指标。
    单看生成图好不好看是主观的，这个数是可以摆出来的证据。
    """
    flat = images.flatten(1)
    n = flat.size(0)
    idx_a = torch.randint(0, n, (max_pairs,), device=flat.device)
    idx_b = torch.randint(0, n, (max_pairs,), device=flat.device)
    keep = idx_a != idx_b                       # 排除自己和自己配对
    return (flat[idx_a[keep]] - flat[idx_b[keep]]).norm(dim=1).mean().item()


if __name__ == "__main__":
    gen, disc = Generator(), Discriminator()
    gen.apply(init_weights)
    z = torch.randn(2, gen.latent_dim)
    fake = gen(z)
    print("生成器输出:", tuple(fake.shape), "范围", (fake.min().item(), fake.max().item()))
    print("判别器输出:", tuple(disc(fake).shape))
    print("生成器参数量: %.1f M" % (sum(p.numel() for p in gen.parameters()) / 1e6))
    print("判别器参数量: %.1f M" % (sum(p.numel() for p in disc.parameters()) / 1e6))
