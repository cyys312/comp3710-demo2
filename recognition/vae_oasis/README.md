# Task 1 — Variational Autoencoder on OASIS Brain MRI

> COMP3710 Lab 2, Part 4 · Easy difficulty (最高 3/7 分)

## 问题描述

对 Preprocessed OASIS 脑部 MR 图像训练一个变分自编码器 (VAE)，
并可视化其隐空间所构成的流形 (manifold)。

## 算法

VAE 由编码器 `q(z|x)`、重参数化采样和解码器 `p(x|z)` 组成，
训练目标为 ELBO 的负值：

```
loss = BCE(重建, 原图) + beta * KL( q(z|x) || N(0, I) )
```

- **重建项**让解码结果贴近输入；
- **KL 项**把后验分布拉向标准正态先验，使隐空间连续、可采样——
  这正是能把它当作「流形」来插值和可视化的原因。
- 重参数化技巧 `z = mu + sigma * eps` 让随机采样这一步仍然可以反向传播。

网络：4 层 stride=2 卷积把 128×128 压到 8×8，再全连接输出 `mu` / `logvar`；
解码器用对称的转置卷积还原，最后接 sigmoid 输出 [0,1]。

## 数据

| 划分 | 目录 |
|---|---|
| train | `keras_png_slices_train/` |
| validate | `keras_png_slices_validate/` |
| test | `keras_png_slices_test/` |

数据集本身已按病例划分好，不重新切分，避免同一病例的相邻切片同时出现在训练集和测试集造成信息泄漏。
VAE 是无监督的，`keras_png_slices_seg_*` 标签不参与训练。

预处理：转灰度、resize 到 128×128、像素归一化到 [0,1]。

## 运行

```bash
python train.py --epochs 30 --latent-dim 32
python predict.py --mode all
```

## 结果

TODO: 训练完成后填入

| 项目 | 数值 |
|---|---|
| latent_dim | |
| 最佳验证 ELBO | |
| 训练时长 | |

图（`outputs/`）：

- `loss_curve.png` — 训练/验证损失
- `reconstruction.png` — 上排原图，下排重建
- `manifold_grid.png` — latent_dim=2 时的 2D 流形网格
- `manifold_umap.png` — 高维隐空间的 UMAP 投影

TODO: 对流形图做一句话解读——相邻位置的脑图是否平滑过渡？有没有明显的聚类结构？

## Demo 要点

- 解释 KL 项的作用，以及去掉它会发生什么（退化成普通 autoencoder，隐空间不连续）
- 解释为什么用重参数化技巧
- 说明 latent_dim 的取舍：2 维可直接画流形但重建糊，32 维清晰但需 UMAP

## AI usage

TODO: 记录使用的模型、用途和提示词要点（任务书要求注明来源）。
