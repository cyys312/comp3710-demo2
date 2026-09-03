# Task 3 — GAN Generation of OASIS Brain MRI

> COMP3710 Lab 2, Part 4 · Hard difficulty（与 Task 1、2 合计最高 7/7 分）

## 问题描述

用生成对抗网络（GAN）生成 Preprocessed OASIS 脑部 MR 图像。任务书要求：

> Images must be suitably realistic and evidence of training (generated images,
> training loss plots etc.) of the models must be provided. […] Results obtained
> must look like unique brains for full marks and **issues such as mode collapse
> need to be fully resolved**.

并且明确警告 GAN «have very chaotic convergence»。因此本实现的重点不只是
"能生成图"，而是**能证明没有发生 mode collapse**。

## 算法

GAN 是生成器 G 与判别器 D 的极小极大博弈：D 学着分辨真假，G 学着骗过 D。
两者交替更新，理论上的均衡点是 D 完全无法分辨（输出恒为 0.5，
对应 BCE 损失 ln 4 ≈ 1.386）。

- **生成器**：从 128 维隐向量出发，4×4 转置卷积展开成 4×4 特征图，
  再经 5 次 stride=2 转置卷积翻倍到 128×128。BatchNorm + ReLU，输出层 Tanh。
  约 13.2 M 参数。
- **判别器**：镜像结构，5 次 stride=2 卷积下采样到 4×4，再用 4×4 卷积输出一个 logit。
  LeakyReLU(0.2)，约 11.2 M 参数。

## 针对 mode collapse 的四项措施

任务书专门点名 mode collapse，所以这里不是朴素 DCGAN：

| 措施 | 作用 |
|---|---|
| **R1 梯度惩罚** | 惩罚判别器在**真实样本处**梯度的平方范数，把它约束成局部 Lipschitz。判别器一旦过强，生成器收到的梯度就会消失，训练随即崩掉。相比 WGAN-GP 不需要在真假样本间插值，开销更小；采用惰性正则，每 16 步做一次 |
| **生成器权重 EMA** | 采样时用指数滑动平均后的权重而非当前权重。提升样本质量最省事的手段之一，几乎没有副作用 |
| **TTUR** | 判别器学习率 3e-4 略高于生成器的 2e-4，让 D 保持"稍强但不过强" |
| **标签平滑** | 真实标签用 0.9 而非 1.0，不让判别器把置信度推进饱和区（饱和后梯度≈0） |

另外两处细节：

- **判别器不用 BatchNorm，改用 InstanceNorm**。BatchNorm 会让同一 batch 内的样本
  通过统计量互相耦合，是 GAN 里已知的不稳定来源。
- **非饱和损失**：生成器最大化 log D(G(z))，而不是最小化 log(1 − D(G(z)))。
  后者在训练早期判别器很强时梯度几乎为 0 —— 这是原始 GAN 论文就指出的问题。

## 数据

只用训练集的 9664 张原图（GAN 是无监督的，`keras_png_slices_seg_*` 标签不参与）。
沿用数据集自带划分。

预处理与 VAE 有一处关键差别：**像素归一化到 [−1, 1] 而不是 [0, 1]**，
因为生成器最后一层是 Tanh。真假数据的取值范围必须一致，
否则判别器只要看数值范围就能分辨，训练会立刻退化。

## 怎么证明没有 mode collapse

单看生成图"好不好看"是主观的。本项目用一个可以摆出数字的指标：

> **多样性 = 样本两两之间的平均 L2 距离**（`modules.diversity_score`），
> 并与**真实数据的同一指标**做比值。

崩塌的生成器会把不同的 z 都映射到少数几个模式上，样本彼此近乎相同，
这个数会远低于真实数据。训练脚本逐轮记录该指标并画成
`outputs/diversity_curve.png`，与真实数据的基准线并排显示。

判定阈值取 70%。

### 训练过程中的多样性变化

| epoch | 生成 | 真实 | 比值 |
|---|---|---|---|
| 1 | 12.36 | 30.96 | 40 % |
| 6 | 7.64 | 31.56 | 24 %（最低点） |
| 10 | 13.56 | 30.62 | 44 % |
| 15 | 19.21 | 29.58 | 65 % |
| 43 | 26.94 | 29.95 | **90 %** |
| 45 | 26.49 | 31.55 | **84 %** |

曲线呈 **V 形**，这是 GAN 早期的正常模式，值得在 demo 时解释：
生成器一开始最省力的策略是输出所有脑图的"平均"——单张看起来像个模糊的脑，
但彼此几乎一样，所以多样性反而先跌到 24 %。等判别器学会识破这种平均图之后，
生成器被迫去覆盖真实分布的各个区域，多样性才开始爬升。
**第 6 轮的低点不是 mode collapse，而是尚未学会多样性**——
两者的区别在于后续是否能自行恢复。

同期的损失也印证了这一点：判别器损失稳定在 1.43–1.48（接近但略高于均衡值 ln 4 ≈ 1.386），
生成器损失从 1.20 降到 0.85，没有出现任何一方"赢死"另一方的发散。

## 结果

训练环境：Rangpur `a100` 分区，NVIDIA A100-PCIE-40GB，作业 `581557`。
128×128 分辨率，batch 64，约 6–7 秒/epoch。

图（`outputs/`）：

- `samples_epoch{005..080}.png` — **固定噪声**下的生成样本随训练的演变。
  用固定 z 是为了让不同 epoch 之间可比：看到的变化全部来自模型本身。
- `loss_curve.png` — 判别器/生成器损失
- `diversity_curve.png` — 多样性指标 vs 真实数据基准（上表的图形版）
- `generated_samples.png` — 最终模型的生成网格（`predict.py` 产出）
- `interpolation.png` — 隐空间线性插值

**样本质量的演变**：第 5 轮还是模糊的椭圆团块；第 15 轮已能认出颅骨轮廓与
脑室的蝶形结构，但噪点明显；第 45 轮噪点大幅减少，脑室边界锐利，
且各样本的脑室大小与形状差异显著。

## 运行

```bash
# 集群（推荐，约 6-7 秒/epoch）
sbatch slurm/gan.slurm
sbatch --export=ALL,ONLY=gan slurm/predict.slurm

# 本机
python recognition/gan_oasis/train.py --epochs 80 --batch-size 64
python recognition/gan_oasis/predict.py --n 64
```

## Demo 要点

- **重点讲多样性指标**：这是"mode collapse 已解决"的客观证据，
  也是本任务与"随便训个 DCGAN"的区别所在。要能解释 V 形曲线的成因。
- 解释四项稳定化措施各自针对什么失效模式（尤其 R1 与非饱和损失）
- 解释为什么判别器要用 InstanceNorm 而非 BatchNorm
- 解释为什么数据要归一化到 [−1, 1] 而不是 [0, 1]
- 用 `interpolation.png` 佐证：健康的生成器在隐空间插值时应平滑过渡，
  崩塌的生成器会在几个模式之间突变
- 与 Task 1 的 VAE 对比：VAE 有显式的似然下界可以监控（ELBO 单调下降就是在进步），
  GAN 没有——损失值本身不能说明生成质量好坏，这正是 GAN 难调的根本原因

## AI usage

见仓库根目录的 [`AI_PROMPTS.md`](../../AI_PROMPTS.md)。
多样性指标与 V 形曲线的解释是自己设计与分析的；
四项稳定化措施的选取依据是任务书对 mode collapse 的明确要求。
