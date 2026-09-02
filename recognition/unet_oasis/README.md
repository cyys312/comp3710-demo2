# Task 2 — UNet Segmentation of OASIS Brain MRI

> COMP3710 Lab 2, Part 4 · Medium difficulty（与 Task 1 合计最高 5/7 分）

## 问题描述

用 UNet 对 Preprocessed OASIS 脑部 MR 图像做分割，
**所有标签的 DSC（Dice Similarity Coefficient）需 > 0.9**，输出必须是 one-hot / categorical 形式。

## 算法

UNet 是编码器-解码器结构，核心是 **skip connection（跳跃连接）**：
编码器每一层的高分辨率特征被拼接到解码器的对应层。
下采样让网络看到大范围上下文（是哪块组织），跳跃连接把丢失的空间细节补回来（边界在哪里），
两者结合才能得到既语义正确又边界锐利的分割结果。

- 编码器：4 级 `DoubleConv` + MaxPool，通道 32→64→128→256，bottleneck 512
- 解码器：转置卷积上采样 + 与编码器特征 concat + `DoubleConv`
- 输出头：1×1 卷积 → 4 通道 logits，配 one-hot 标签

损失函数 = 0.5 × CrossEntropy + 0.5 × Dice Loss。
纯交叉熵在类别不平衡（背景占比大）时会偏向背景；Dice Loss 直接优化评测指标本身。

## 数据

| 划分 | 原图 | 标签 |
|---|---|---|
| train | `keras_png_slices_train/` | `keras_png_slices_seg_train/` |
| validate | `keras_png_slices_validate/` | `keras_png_slices_seg_validate/` |
| test | `keras_png_slices_test/` | `keras_png_slices_seg_test/` |

沿用数据集自带划分（按病例分开，避免相邻切片泄漏）。
原图与标签按文件名编号配对：`case_441_slice_0` ↔ `seg_441_slice_0`。

预处理：resize 到 256×256（**标签必须用最近邻插值**，否则会插出不存在的中间类别值）；
标签灰度 {0, 85, 170, 255} 映射为类别索引 {0,1,2,3}，训练时转 one-hot。

## 运行

本机（Apple MPS / CPU 自动识别）：

```bash
python recognition/unet_oasis/train.py --epochs 25 --batch-size 16
python recognition/unet_oasis/predict.py --n-show 4      # demo 现场跑这个
```

Rangpur 集群（数据在 `/home/groups/comp3710/OASIS`，无需上传）：

```bash
sbatch slurm/unet.slurm
sbatch --export=ALL,ONLY=unet slurm/predict.slurm
```

## 结果

训练环境：Rangpur `a100` 分区，NVIDIA A100-PCIE-40GB（作业 581337）。
25 个 epoch，约 0.4 分钟/epoch，共约 10 分钟。
下表是在 **544 张测试集切片**上的结果（该划分完全没有参与训练与调参）：

| 类别 | 组织 | 像素占比 | 测试集 DSC |
|---|---|---|---|
| 0 | 背景 | 72.30 % | **0.9993** |
| 1 | 脑脊液 CSF | 5.66 % | **0.9651** |
| 2 | 灰质 | 11.39 % | **0.9658** |
| 3 | 白质 | 10.65 % | **0.9795** |
| **mean** | | | **0.9774** |

**全部四个标签均 > 0.9，满足任务书要求。** 最好的 checkpoint 来自第 22 个 epoch
（按验证集平均 DSC 选取，测试集只在最后评估一次）。

### 标签与组织的对应关系是怎么确定的

数据集本身没有给标签字典。我用「各标签区域在原图中的平均灰度」反推：

| 标签 | 平均灰度 |
|---|---|
| c0 | 0.0551 |
| c1 | 0.1407 |
| c2 | 0.3153 |
| c3 | 0.4755 |

灰度严格单调递增，与 T1 加权 MRI 的组织对比度顺序完全一致：
背景最暗 → 脑脊液（T1 上低信号）→ 灰质 → 白质最亮。

### 为什么 c1 的 DSC 最低

CSF 只占 5.66 % 的像素，且多为脑室边缘的细长结构，
边界像素占其总面积的比例最高 —— Dice 对小目标的边界误差最敏感，
错一圈像素对小结构的惩罚远大于对大结构。这也是用 CE + Dice
组合损失而非纯 CE 的原因：背景占 72.30 %，纯 CE 会让模型偏向背景。

图（`outputs/`）：

- `dice_curve.png` — 逐类 DSC 随 epoch 变化（含 0.9 参考线）
- `segmentation_examples.png` — 原图 / 真值 / 预测三列对比

## Demo 要点

- 现场对测试集跑 `predict.py` 展示推理结果（任务书明确要求）
- 解释 skip connection 为什么对分割精度是关键
- 解释每一类 DSC 的差异从何而来（小结构类别更难，边界像素占比高）
- 说明为什么标签 resize 必须用最近邻

## AI usage

见仓库根目录的 [`AI_PROMPTS.md`](../../AI_PROMPTS.md)。与本子项目直接相关的两处：
AI 初版把软 Dice 同时当作损失和评测指标（评测必须用硬预测），
以及把 batch 级 DSC 取平均当作数据集级 DSC（比值不能这样平均）——
两处都已修正，理由写在 `modules.py` 的注释里。
