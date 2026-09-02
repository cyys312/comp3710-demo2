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

```bash
python train.py --epochs 20 --batch-size 16
python predict.py --n-show 4      # demo 现场跑这个
```

## 结果

TODO: 训练完成后填入

| 类别 | 含义 | 测试集 DSC |
|---|---|---|
| 0 | 背景 | |
| 1 | | |
| 2 | | |
| 3 | | |
| **mean** | | |

图（`outputs/`）：

- `loss_curve.png` — 训练损失
- `dice_curve.png` — 逐类 DSC 随 epoch 变化（含 0.9 参考线）
- `segmentation_examples.png` — 原图 / 真值 / 预测三列对比

## Demo 要点

- 现场对测试集跑 `predict.py` 展示推理结果（任务书明确要求）
- 解释 skip connection 为什么对分割精度是关键
- 解释每一类 DSC 的差异从何而来（小结构类别更难，边界像素占比高）
- 说明为什么标签 resize 必须用最近邻

## AI usage

TODO: 记录使用的模型、用途和提示词要点（任务书要求注明来源）。
