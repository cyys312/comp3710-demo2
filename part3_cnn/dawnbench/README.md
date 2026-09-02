# Part 3.2 — DAWNBench: ResNet-18 on CIFAR-10

> COMP3710 Lab 2, Part 3.2（4 分）

## 任务书要求

1. 准确率 > 90 %，且训练时间"在集群上通常 30 分钟以内"（1 分）
2. 演示时能在 Rangpur 上跑推理与一个 epoch 的训练（1 分）
3. 用混合精度达到 94 %，时间与 V100 上的约 360 秒相当或更快（2 分）

不允许使用预训练模型。

## 结果

Rangpur `a100` 分区，NVIDIA A100-PCIE-40GB，作业 `581349`：

| 指标 | 数值 |
|---|---|
| 测试集准确率 | **94.15 %** |
| 总训练时间（30 epoch，含每轮评估） | **95 秒** |
| 单个 epoch 训练时间 | 1.9 秒 |

| 要求 | 判定 |
|---|---|
| >90 % 且 <30 分钟 | **达标** |
| ≥94 % 且 ≤360 秒（V100 基准） | **达标**（95 秒，快约 3.8 倍） |

## 模型

`modules.py` —— 为 32×32 输入改造的 ResNet-18，从零训练：

- 首层用 3×3 stride=1 卷积代替 ImageNet 版的 7×7 stride=2，并**去掉 maxpool**。
  32×32 的输入经不起开头就做 4 倍下采样，否则进入 layer1 时只剩 8×8。
- 4 个 stage 各 2 个 BasicBlock，通道 64→128→256→512，尺寸或通道变化时
  捷径用 1×1 卷积做投影。
- 参数量约 11.2 M。

## 训练配置

- **AMP 混合精度**（`torch.amp`）：A100 的 Tensor Core 上 fp16/bf16 吞吐远高于 fp32
- **OneCycleLR + SGD(nesterov)**：`max_lr=0.4`，`pct_start=0.25`，
  少量 epoch 内快速收敛的标准配方
- **label smoothing 0.1**、**weight decay 5e-4**
- **channels_last** 内存格式 + `cudnn.benchmark`

## 关键优化：把数据管线搬到 GPU

这是能否拿到那 2 分的分水岭，做法是**先测量再优化**。

**测量**：最初用 torchvision 的 `DataLoader` + PIL 增强管线
（`RandomCrop → Flip → ToTensor → Normalize`），在 A100 上
**227 秒都跑不完一个 epoch**（97 步，即 >2.3 秒/步）。
而 ResNet-18 在 A100 上一步的实际计算只有几十毫秒——
**瓶颈在数据管线，不在模型**。

**原因**：Rangpur 的 a100 节点每个只有 8 个 CPU 核（作业申请 4 核），
CPU 侧的 PIL 增强根本喂不饱 GPU，A100 大部分时间在空转。

**做法**（`dataset.GPUCifar`）：CIFAR-10 的原始像素一共只有
50000 × 3 × 32 × 32 = **153 MB**（uint8），相对 40 GB 显存微不足道。
一次性搬进显存后，随机裁剪与翻转全部用张量算子完成，CPU 彻底退出热路径。

**效果**：

| 数据管线 | 单 epoch 耗时 |
|---|---|
| torchvision DataLoader（4 workers） | >227 秒仍未跑完 |
| GPU 常驻 | **1.9 秒** |

**等价性验证**：随机裁剪的实现是先 reflect padding 到 40×40，
再对每个样本独立取 `[0,8]` 的随机偏移裁回 32×32，与
`T.RandomCrop(32, padding=4, padding_mode="reflect")` 逐样本等价。
偏移恰为 (4,4) 时能**精确还原原图**——这一条确认了高级索引没有写错位。
翻转与归一化同理逐条对齐，注释写在 `dataset.py` 里。

两条管线都保留了，用 `--loader {gpu,cpu}` 切换，便于现场演示对比。

## 运行

```bash
# 集群（推荐）
sbatch part3_cnn/dawnbench/run_rangpur.slurm

# 本地
python part3_cnn/dawnbench/train.py --epochs 30 --batch-size 512
python part3_cnn/dawnbench/predict.py

# 演示时对比两条数据管线
python part3_cnn/dawnbench/train.py --epochs 1 --loader cpu
python part3_cnn/dawnbench/train.py --epochs 1 --loader gpu
```

CIFAR-10 在集群上已有共享副本 `/home/groups/cifar/CIFAR-10`，
`run_rangpur.slurm` 通过 `$CIFAR_ROOT` 指过去，**不要现下**
（torchvision 默认源到集群只有约 3 KB/s）。

## Demo 要点

- 现场跑推理与一个 epoch 的训练（任务书第 2 条明确要求，1 分）
- 解释为什么首层要从 7×7 stride=2 改成 3×3 stride=1 并去掉 maxpool
- 解释残差连接解决的是什么问题（深层网络的梯度消失与退化）
- 解释混合精度为什么既快又不掉精度（GradScaler 如何防止 fp16 梯度下溢）
- **重点讲数据管线那段**：先测量、定位瓶颈在 CPU 而非 GPU、再针对性优化，
  以及如何验证优化后的增强与原来等价

## AI usage

见仓库根目录的 [`AI_PROMPTS.md`](../../AI_PROMPTS.md)。
AI 给的初版用的是 CPU DataLoader；GPU 常驻管线是在测出瓶颈之后自己加的，
等价性验证也是自己设计的。
