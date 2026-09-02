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

本机（Apple MPS / CPU 自动识别）：

```bash
python recognition/vae_oasis/train.py --epochs 40 --latent-dim 2    # 画流形网格用
python recognition/vae_oasis/train.py --epochs 40 --latent-dim 32   # 重建更清晰
python recognition/vae_oasis/predict.py --mode all --ckpt recognition/vae_oasis/checkpoints/vae_z2.pth
```

Rangpur 集群（数据在 `/home/groups/comp3710/OASIS`，无需上传）：

```bash
sbatch --export=ALL,LATENT_DIM=2  --job-name=vae-z2  slurm/vae.slurm
sbatch --export=ALL,LATENT_DIM=32 --job-name=vae-z32 slurm/vae.slurm
sbatch --export=ALL,ONLY=vae slurm/predict.slurm
```

## 结果

训练环境：Rangpur `a100` 分区，NVIDIA A100-PCIE-40GB。各训练 40 个 epoch，
约 6 秒/epoch。为了兼顾「能直接画流形」与「重建清晰」，训练了两个模型：

| latent_dim | 最佳验证 ELBO | 重建项 | KL 项 | 训练时长 | 作业号 |
|---|---|---|---|---|---|
| 2 | 4265.93 | 4173.72 | 8.24 | 4 分 43 秒 | 581351 |
| 32 | **4183.05** | 4053.01 | 32.46 | 4 分 47 秒 | 581352 |

（损失为每张图的负 ELBO，重建项是 128×128 像素上求和的 BCE，故数值在数千量级。）

**怎么读这两行**：32 维的 ELBO 更低（重建更好），代价是隐空间维度太高、
没法直接可视化；2 维重建略差，但可以在隐平面上铺网格直接把流形画出来。
KL 项也符合预期：2 维时 8.24 nats ≈ 4.1 nats/维，32 维时 32.46 nats ≈ 1.0 nats/维 ——
维度越多，每一维承载的信息越少，但都远离 0，说明**没有发生后验坍缩**
（若 KL→0 则隐变量被忽略，模型退化成普通 autoencoder）。

### 流形可视化

`z2_manifold_grid.png` 是核心产物：在 latent 平面上按标准正态的**分位数**
（而非等距）铺 20×20 网格并逐点解码。用分位数是因为先验是标准正态，
等分位采样才能让网格均匀覆盖概率质量，边缘不会全落在训练时没见过的区域。

读图结论：网格上**相邻位置的脑图是平滑过渡的**，中央脑室的蝶形结构沿一个方向
连续地由窄变宽、由尖变圆，没有突变或撕裂。这正是 KL 项的作用——它把后验拉向
标准正态，逼迫隐空间"填满"而不是退化成一堆互相孤立的点，
因此在两个训练样本之间插值也能解出合理的脑图。

`z32_manifold_umap.png` 是 32 维隐空间的 2D 投影。集群的 conda 环境里没有
umap-learn，代码会自动退回 PCA（纯 torch SVD 实现，见 `pca_2d`）。
对「隐空间是否连续」这个问题线性投影已足够；UMAP 的长处在于保留非线性邻域结构，
有则更好，没有不影响结论。

图（`outputs/`，按 latent_dim 加前缀）：

- `z*_loss_curve.png` — 训练/验证损失
- `z*_reconstruction.png` — 上排原图，下排重建
- `z2_manifold_grid.png` — **2D 流形网格（任务书要求的可视化）**
- `z*_manifold_umap.png` — 隐空间的 2D 投影（UMAP，缺失时自动退回 PCA）
- `z*_recon_epoch{05..40}.png` — 重建质量随训练的演变

## Demo 要点

- 解释 KL 项的作用，以及去掉它会发生什么（退化成普通 autoencoder，隐空间不连续）
- 解释为什么用重参数化技巧
- 说明 latent_dim 的取舍：2 维可直接画流形但重建糊，32 维清晰但需 UMAP

## AI usage

见仓库根目录的 [`AI_PROMPTS.md`](../../AI_PROMPTS.md)。与本子项目相关的改动：
去掉了对 scipy 的依赖（`normal_ppf` 改用 torch 的 `erfinv` 实现，与
`scipy.stats.norm.ppf` 的偏差为 8.9e-16），以及给 UMAP 加了 PCA 兜底 ——
集群环境两者都没有，为一个函数装一整个依赖不划算。
