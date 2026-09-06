# COMP3710 — Pattern Analysis, Lab Demonstration 2

Semester 2, 2026 · The University of Queensland

仓库：<https://github.com/cyys312/comp3710-demo2>

本仓库包含 COMP3710 Lab 2（Pattern Recognition）四个部分的实现。
框架统一使用 **PyTorch**。

---

## 目录结构

```
comp3710/
├── data/                       # 数据集（不入 git），见下方「数据」
├── docs/                       # 任务书与评分标准 PDF
├── part1_dft/                  # Part 1 — 离散傅里叶变换 (1 分)
│   └── dft.py
├── part2_eigenfaces/           # Part 2 — Eigenfaces / PCA + 随机森林 (1 分)
│   └── eigenfaces.py
├── part3_cnn/                  # Part 3 — CNN (5 分)
│   ├── cnn_lfw.py              #   3.1 LFW 人脸 CNN 分类器 (1 分)
│   └── dawnbench/              #   3.2 DAWNBench: ResNet-18 + CIFAR-10 (4 分)
│       ├── modules.py          #       模型定义
│       ├── dataset.py          #       数据加载与增强
│       ├── train.py            #       训练（混合精度 + OneCycle）
│       ├── predict.py          #       推理与评估
│       └── run_rangpur.slurm   #       Rangpur 集群提交脚本
├── recognition/                # Part 4 — 识别任务 (7 分)
│   ├── vae_oasis/              #   Task 1 — VAE (Easy, 3 分)
│   ├── unet_oasis/             #   Task 2 — UNet 分割 (Medium, 累计 5 分)
│   └── gan_oasis/              #   Task 3 — GAN 生成脑图 (Hard, 累计 7 分)
├── slurm/                      # Rangpur 作业脚本（参数经集群实测核对）
├── AI_PROMPTS.md               # AI 使用记录（评分标准要求）
├── DEMO_PREP.md                # 演示准备：预期问题与答法
├── CODE_WALKTHROUGH.md         # 代码讲解：每个文件在干嘛、逐块拆解
├── requirements.txt
└── README.md
```

每个 `recognition/` 子项目遵循课程约定的四件套：

| 文件 | 职责 |
|---|---|
| `modules.py` | 模型/网络各组件的定义 |
| `dataset.py` | 数据加载器与预处理 |
| `train.py` | 训练、验证、保存、损失曲线 |
| `predict.py` | 载入 checkpoint 做推理与可视化 |
| `README.md` | 算法原理、数据划分、结果与图 |

---

## 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU 版 PyTorch 请按 <https://pytorch.org> 上对应 CUDA 版本的命令安装。

## 数据

- **LFW（Part 2/3.1）**：由 `sklearn.datasets.fetch_lfw_people` 首次运行时自动下载。
- **CIFAR-10（Part 3.2）**：本地由 `torchvision.datasets.CIFAR10` 自动下载到 `data/`。
  **集群上不要现下**——已有共享副本 `/home/groups/cifar/CIFAR-10`
  （`cifar-10-batches-py` 格式，torchvision 可直接读）；
  torchvision 默认源到集群只有约 3 KB/s，170 MB 要下十几个小时。
- **OASIS 脑部 MRI（Part 4）**：本地解压到 `data/keras_png_slices_data/`；
  集群上位于 `/home/groups/comp3710/OASIS`，无需上传。

```bash
unzip keras_png_slices_data.zip -d data/
```

目录含 6 个子集：`keras_png_slices_{train,validate,test}`（原图）
与 `keras_png_slices_seg_{train,validate,test}`（分割标签）。

## 运行

```bash
python part1_dft/dft.py
python part2_eigenfaces/eigenfaces.py
python part3_cnn/cnn_lfw.py
python part3_cnn/dawnbench/train.py --epochs 30
python recognition/vae_oasis/train.py
python recognition/unet_oasis/train.py
```

## 在 Rangpur 上训练

Part 3.2（DAWNBench）必须在集群上跑，Part 4 的 VAE/UNet 在集群上也快得多
（UNet 一个 epoch：A100 约 0.4 分钟 vs Apple M5 约 7 分钟）。

```bash
# 1) 首次拉取代码（校外需先连 UQ VPN，否则 ssh 22 端口超时）
ssh <uqusername>@rangpur.compute.eait.uq.edu.au
git clone https://github.com/cyys312/comp3710-demo2.git
cd comp3710-demo2

# 2) 提交作业。所有数据集集群上都已有，不需要上传或下载。
sbatch part3_cnn/dawnbench/run_rangpur.slurm     # Part 3.2
sbatch slurm/unet.slurm                          # Task 2
sbatch --export=ALL,LATENT_DIM=2 --job-name=vae-z2 slurm/vae.slurm   # Task 1
sbatch slurm/predict.slurm                       # 推理与可视化（demo 现场跑这个）

squeue -u $USER
tail -f logs/<jobname>-<jobid>.out
```

### 集群配置的三个坑（实测核对，照抄模板会挂）

用 `sinfo` / `scontrol show node` / `sacctmgr` 核对后发现，网上常见的 slurm
模板在 Rangpur 上会直接失败：

| 常见写法 | 实际情况 | 后果 |
|---|---|---|
| `#SBATCH --mem=32G` | 所有 a100 节点 `RealMemory=1` | **作业无限期 PENDING**——这是很多人卡住的真正原因 |
| `--partition=comp3710` | 该分区 `AllowAccounts=comp3710`，账号关联未开通 | 一律 `PENDING (PartitionConfig)`，改用 `a100` 分区 |
| `module load cuda` | 只有 cuda/11.1、11.4、12.2，而 conda 环境里是 torch 2.13.0+**cu130**（自带运行时） | 版本冲突风险，正确做法是不加载 |

另外 **登录节点（login0）不要跑任何计算任务**，包括下载数据集——
那是全课共享的机器。需要下载就提交一个 `--partition=cpu` 的作业。
所有细节写在 `slurm/_common.sh` 的注释里。

## 结果总览

所有数字都是实测的，命令与日志见各子目录 README。

| 部分 | 分值 | 结果 | 状态 |
|---|---|---|---|
| Part 1 — DFT | 1 | 四种实现耗时对比，规模扫描（见下表） | 完成 |
| Part 2 — Eigenfaces + 随机森林 | 1 | 准确率 0.5776（基线 0.4130）；加类别权重后 0.7298 | 完成 |
| Part 3.1 — LFW CNN | 1 | 准确率 **0.8509**，显著优于 Part 2 | 完成 |
| Part 3.2 — DAWNBench | 4 | **94.15 % / 95 秒**（A100） | 完成 |
| Part 4.1 — edX 进阶 Git 短课程 | 1 | — | **待完成** |
| Part 4 Task 1 — VAE + 流形 | — | 验证 ELBO 4183.05，2D 流形网格已出 | 完成 |
| Part 4 Task 2 — UNet 分割 | — | 测试集 **mean DSC 0.9774**，四类全部 > 0.9 | 完成 |
| Part 4 Task 3 — GAN | — | 测试集多样性达真实数据 **88 %**，无 mode collapse | 完成 |

### Part 1 — 四种 DFT 实现的耗时（秒，取 3 次最小值）

| 实现 | N=1024 | N=2048 | N=4096 | 复杂度 |
|---|---|---|---|---|
| NumPy 朴素 DFT（矩阵形式） | 0.009265 | 0.037120 | 0.146261 | O(N²) |
| PyTorch 朴素 DFT（GPU） | 0.002785 | 0.006693 | 0.010211 | O(N²) |
| NumPy FFT | 0.000006 | 0.000011 | 0.000019 | O(N log N) |
| PyTorch FFT（GPU） | 0.000269 | 0.000236 | 0.000374 | O(N log N) |

排序（快→慢）：**NumPy FFT < PyTorch FFT(GPU) < PyTorch 朴素 DFT(GPU) < NumPy 朴素 DFT**。

三点解释：

1. **FFT 最快是因为算法，不是因为硬件**。N 每翻一倍，朴素 DFT 的耗时约变 4 倍
   （0.0093 → 0.037 → 0.146，正是 O(N²)），FFT 只增长不到 2 倍。
   算法复杂度的差距，堆多少并行核心都补不回来。
2. **GPU 让朴素 DFT 快了一个数量级但改不了复杂度**。N=4096 时 GPU 版比 CPU 版快
   14 倍（0.0102 vs 0.1463），因为 N² 次乘加被摊到数千个核心上；
   但它仍是 O(N²)，N 再翻几倍照样会输给 CPU 上的 FFT。
3. **GPU 的 FFT 反而不如 NumPy 的**，因为这个规模下每次 kernel 启动的固定开销
   （约 0.25 ms）就超过了计算本身。GPU 要在计算量足够大时才划算。

补充：同为 O(N²)，NumPy 的矩阵形式比教科书式的双重 Python 循环快约 25 倍
（N=512 时 0.0023 s vs 0.0581 s）——**复杂度和常数因子是两件事**。

### 待完成

- [ ] **Part 4.1 — edX 进阶 Git 短课程**（1 分，纯粹花时间，不写代码）

Task 1、2、3 全部实现，Part 4 按 **Hard 难度**计，任务部分上限 7/7。

### 关于 Task 3 的 mode collapse

任务书要求 mode collapse «need to be fully resolved»。本项目不靠肉眼判断，
而是用**样本两两平均 L2 距离**作为多样性指标，与真实数据的同一指标做比值：

| epoch | 6 | 15 | 43 | 80（最终） |
|---|---|---|---|---|
| 多样性 / 真实数据 | 24 % | 65 % | 90 % | **84 %** |

`predict.py` 用测试集（544 张，未参与训练）作基准独立复核为 **88 %**，结论一致。

曲线呈 V 形是 GAN 早期的正常现象——生成器先学"平均脑"（像脑但彼此雷同），
判别器识破后才被迫覆盖真实分布。**低点不是崩塌，区别在于能否自行恢复**。
详见 [`recognition/gan_oasis/README.md`](recognition/gan_oasis/README.md)。

## AI 使用声明

本仓库部分代码在 **Claude Code (Opus 5)** 协助下编写，符合任务书
"Use of Artificial Intelligence" 一节与评分标准 "Fair AI Usage" 的要求。

完整记录见 [`AI_PROMPTS.md`](AI_PROMPTS.md)，其中列出了 AI 初版代码中
**被我发现并修正的 6 处问题**（含两处会让上报的 DSC 失真的错误、
三处会导致集群作业永远排不上的 slurm 参数），以及我自己补做的对照实验。
