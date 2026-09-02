# AI 使用记录 (AI Usage Log)

课程评分标准要求：若使用 AI 工具，需说明用法与推理过程，并可能被要求提供
prompt 历史（`prompt_history.pdf` 或等价材料）。本文件记录本仓库开发过程中
AI 的实际参与方式、我对 AI 输出的**验证与修正**，以及最终由我承担的判断。

工具：**Claude Code (Opus 5)**，在本机终端与 Rangpur 集群上交互式使用。
时间：2026-09-02。

---

## 1. AI 的用法定位

评分标准写明「把题目直接抄给 AI、再把答案抄回来」不算合格用法。本项目里 AI 的
角色是**加速实现与排障**，而设计决策、结果解释、正确性验证由我把关。下面按
「AI 产出 → 我发现的问题 → 修正」的顺序记录，这些修正本身就是理解的证据。

---

## 2. AI 初版代码中被我发现并修正的问题

### 2.1 Part 1 漏掉了任务书的核心要求

AI 最初写的 `dft.py` 只有 NumPy 版的朴素 DFT 与 `np.fft.fft` 对比。
重读任务书第 5 页发现明确要求：

> modify the 'square_wave', 'square_wave_fourier' and 'naive_dft' functions so that
> they are implemented using TensorFlow (TF) or PyTorch operations. For 'naive_dft'
> in particular, create a second version that explicitly runs on the GPU

即必须有 **PyTorch 版**，且 `naive_dft` 要有**显式跑在 GPU 上**的版本。
修正：补齐 `square_wave_torch` / `square_wave_fourier_torch` / `naive_dft_torch`，
并加了规模扫描的计时基准。另外补了一个真正的双重循环版 `naive_dft_loops`，
用来说明「同为 O(N²)，实现方式差 25 倍」——这一点任务书没要求，但它是理解
「为什么最快的最快」的关键：**算法复杂度和常数因子是两件事**。

### 2.2 DSC 用了软 Dice，会让上报的分数失真

AI 写的 `dice_per_class` 直接对 softmax 概率算 Dice，训练损失和评测指标共用同一个
函数。这是错的：任务书要求的 DSC 是对**硬预测**（argmax 之后的类别图）算的。
软 Dice 在模型不自信时偏低、过分自信时偏高，拿它上报会失真。

修正：拆成两个函数——`dice_per_class`（软，可导，只给损失用）与
`hard_dice_counts`（硬，只给评测用）。

### 2.3 DSC 的聚合方式也是错的

原实现是「每个 batch 算一个 DSC，再除以 batch 数」。DSC 是比值，
**batch 级 DSC 的平均 ≠ 数据集级 DSC**，而且最后一个不满的 batch 被赋予了同等权重。

修正：`hard_dice_counts` 返回逐类的交集与基数计数，跨 batch 累加完再做一次除法
（见 `recognition/unet_oasis/modules.py` 的注释）。

### 2.4 slurm 脚本的参数是猜的，会导致作业永远排不上

AI 写的 `run_rangpur.slurm` 里 `--partition=a100`、`module load cuda`、`--mem=32G`
都是按常见模板猜的。在集群上用 `sinfo` / `scontrol` / `sacctmgr` 逐条核对后发现三处错：

| 猜测 | 实测 | 后果 |
|---|---|---|
| `--mem=32G` | 所有 a100 节点 `RealMemory=1` | **作业无限期 PENDING**，这是当时全课很多人卡住的真正原因 |
| `--partition=comp3710` | 该分区 `AllowAccounts=comp3710`，账号关联未开通 | 一律 `PENDING (PartitionConfig)`；改用 `a100` 分区 |
| `module load cuda` | 只有 cuda/11.1、11.4、12.2，而 conda 环境里是 torch 2.13.0+**cu130**（自带运行时） | 加载反而可能冲突，正确做法是不加载 |

这一条说明 AI 对**具体集群的配置**是无知的，只能给模板；必须自己去核对。

### 2.5 多进程 DataLoader 在本机是负优化

AI 默认写了 `num_workers=4/8` + `persistent_workers=True`。在 Apple Silicon 上实测：

| 配置 | 耗时 |
|---|---|
| `num_workers=0` | **9 ms/batch** |
| `num_workers=6, persistent=True` | 262 ms/batch |
| `num_workers=6, persistent=False` | 3249 ms/batch |

单张 OASIS 切片解码只要约 1 ms，多进程的 IPC 序列化开销远大于收益，**慢 29 倍**；
而且 fork 与 MPS 并存时还把主进程卡在过不可中断等待上。
修正：加 `auto_workers()`，按设备选择（CUDA 用 4，MPS/CPU 用 0）。

### 2.6 标签解码为每张图分配了 H×W×4 的临时数组

原实现对每个像素与 4 个参考灰度值求距离再取 argmin。改成 256 项的查找表
（`_LABEL_LUT`），一次索引完成映射。

---

## 3. 我自己加的实验（AI 没有要求做）

### 3.1 Part 2：给 57.76% 这个数字加上参照系

AI 跑出随机森林 57.76% 就停了。这个数字单独看没有意义，我补了两组对照：

- **多数类基线**：LFW 里 George W Bush 一人占测试集 41%，全猜他就有 41.3%。
- **主成分个数消融**：25→0.6304、**50→0.6553**、100→0.5807、150→0.5776。
  成分越多**反而越差**。解释：高阶 eigenface 主要编码光照与姿态噪声；随机森林
  默认 `max_features='sqrt'`，150 维时每次分裂只抽约 12 个特征，抽到噪声的概率更高。
- **类别加权**：`class_weight='balanced'` 把准确率从 0.5776 提到 0.7298，
  macro-F1 从 0.3022 提到 0.6361 —— 混淆矩阵显示原模型几乎把所有样本都判给了多数类。

### 3.2 Part 3.2：先测量再优化

AI 给的 DAWNBench 方案用 torchvision 的 CPU DataLoader。在 A100 上实测
**227 秒都跑不完一个 epoch**（97 步，>2.3 s/step），而 ResNet-18 在 A100 上的
实际计算只要几十毫秒——瓶颈在数据管线，不在模型。

原因：a100 节点每个只有 8 个 CPU 核，PIL 的 `RandomCrop → Flip → ToTensor → Normalize`
喂不饱 GPU。

我的做法：把整个 CIFAR-10（uint8 只有 153 MB）搬进 40 GB 显存，随机裁剪与翻转
改用张量算子在 GPU 上做（`dataset.GPUCifar`）。结果 **1.9 秒/epoch**，
30 轮共 **95 秒达到 94.15%**，比任务书的 V100 基准（360 秒）快近 4 倍。

等价性我做了验证：reflect padding 到 40×40 后取 `[0,8]` 的逐样本随机偏移裁回
32×32，当偏移恰为 (4,4) 时能**精确还原原图**——这确认了索引写法没有错位。

---

## 4. 仍然由我判断、AI 无法代劳的部分

- 任务书三档难度的取舍（Easy/Medium/Hard）与由此决定的分数上限
- 数据划分沿用数据集自带的 train/validate/test（按病例分开），不重新切分 ——
  相邻切片高度相关，随机切分会让同一病例的切片同时进训练集和测试集，造成泄漏
- 标签 resize 必须用最近邻而非双线性，否则会插值出 `{0,85,170,255}` 之外的中间值
- UNet 损失用 CE + Dice 各半的理由（背景占 72.29%，纯 CE 会偏向背景）
- 上述所有实测数字的取得与解释

---

## 5. 可提供的证据

- 本文件（开发过程与修正记录）
- Git commit 历史（按 Part 分组，commit message 说明每次改动的理由）
- 各 `outputs/` 目录下的结果图与 `logs/` 下的完整训练日志
- 集群作业 ID：DAWNBench `581349`、UNet `581337`、VAE `581351`/`581352`
