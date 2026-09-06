# Part 3.2 — DAWNBench: ResNet-18 on CIFAR-10

> COMP3710 Lab 2, Part 3.2 (4 marks)

## Task sheet requirements

1. Accuracy > 90 %, with a training time "typically under 30 minutes on the cluster" (1 mark)
2. Able to run inference and one epoch of training on Rangpur during the demo (1 mark)
3. Reach 94 % using mixed precision, at a time comparable to or faster than the roughly 360 s on a
   V100 (2 marks)

Pretrained models are not allowed.

## Results

Rangpur `a100` partition, NVIDIA A100-PCIE-40GB, job `581349`:

| Metric | Value |
|---|---|
| Test accuracy | **94.15 %** |
| Total training time (30 epochs, evaluation each epoch included) | **95 s** |
| Time for a single training epoch | 1.9 s |

| Requirement | Verdict |
|---|---|
| >90 % and <30 minutes | **met** |
| ≥94 % and ≤360 s (V100 baseline) | **met** (95 s, about 3.8× faster) |

## Model

`modules.py` — ResNet-18 reworked for 32×32 inputs, trained from scratch:

- The first layer is a 3×3 stride=1 convolution in place of the ImageNet version's 7×7 stride=2,
  and the **maxpool is removed**. A 32×32 input cannot survive 4× downsampling right at the start;
  otherwise only 8×8 is left by the time it reaches layer1.
- 4 stages of 2 BasicBlocks each, channels 64→128→256→512; where the spatial size or the channel
  count changes, the shortcut uses a 1×1 convolution as a projection.
- About 11.2 M parameters.

## Training configuration

- **AMP mixed precision** (`torch.amp`): on the A100's Tensor Cores, fp16/bf16 throughput is far
  above fp32
- **OneCycleLR + SGD(nesterov)**: `max_lr=0.4`, `pct_start=0.25`, the standard recipe for fast
  convergence within a small number of epochs
- **label smoothing 0.1**, **weight decay 5e-4**
- **channels_last** memory format + `cudnn.benchmark`

## Key optimisation: moving the data pipeline onto the GPU

This is what decides whether those 2 marks are earned, and the approach was **measure first, then
optimise**.

**Measurement**: the first version used torchvision's `DataLoader` with a PIL augmentation
pipeline (`RandomCrop → Flip → ToTensor → Normalize`), which on the A100 **could not get through a
single epoch in 227 s** (97 steps, i.e. >2.3 s/step). The actual compute of one ResNet-18 step on
an A100 is only tens of milliseconds — **the bottleneck was the data pipeline, not the model**.

**Cause**: each a100 node on Rangpur has only 8 CPU cores (the job requests 4), so the CPU-side
PIL augmentation cannot keep the GPU fed and the A100 spends most of its time idle.

**Approach** (`dataset.GPUCifar`): the raw CIFAR-10 pixels amount to only
50000 × 3 × 32 × 32 = **153 MB** (uint8), negligible against 40 GB of GPU memory. Once they are
moved into GPU memory in one go, random cropping and flipping are done entirely with tensor
operations and the CPU leaves the hot path altogether.

**Effect**:

| Data pipeline | Time per epoch |
|---|---|
| torchvision DataLoader (4 workers) | >227 s and still unfinished |
| GPU-resident | **1.9 s** |

**Equivalence check**: the random crop is implemented by reflect-padding to 40×40 and then taking
an independent random `[0,8]` offset per sample to crop back to 32×32, which is per-sample
equivalent to `T.RandomCrop(32, padding=4, padding_mode="reflect")`. With an offset of exactly
(4,4) it **reproduces the original image exactly** — this check confirms the advanced indexing is
not off by one. Flipping and normalisation were lined up the same way, item by item; the notes are
in `dataset.py`.

Both pipelines are kept and switched with `--loader {gpu,cpu}`, which makes the comparison easy to
show live.

## Running

```bash
# Cluster (recommended)
sbatch part3_cnn/dawnbench/run_rangpur.slurm

# Local
python part3_cnn/dawnbench/train.py --epochs 30 --batch-size 512
python part3_cnn/dawnbench/predict.py

# Compare the two data pipelines during the demo
python part3_cnn/dawnbench/train.py --epochs 1 --loader cpu
python part3_cnn/dawnbench/train.py --epochs 1 --loader gpu
```

CIFAR-10 already has a shared copy on the cluster at `/home/groups/cifar/CIFAR-10`, and
`run_rangpur.slurm` points at it via `$CIFAR_ROOT`. **Do not download it on the spot**
(torchvision's default source reaches the cluster at about 3 KB/s).

## Demo notes

- Run inference and one epoch of training live (explicitly required by item 2 of the task sheet,
  1 mark)
- Explain why the first layer changes from 7×7 stride=2 to 3×3 stride=1 and the maxpool is dropped
- Explain what problem residual connections solve (vanishing gradients and degradation in deep
  networks)
- Explain why mixed precision is fast without costing accuracy (how GradScaler prevents fp16
  gradient underflow)
- **Focus on the data-pipeline section**: measure first, locate the bottleneck on the CPU rather
  than the GPU, then optimise accordingly — and how the optimised augmentation was verified
  equivalent to the original

## AI usage

See [`AI_PROMPTS.md`](../../AI_PROMPTS.md) in the repository root.
The first version from the AI used the CPU DataLoader; the GPU-resident pipeline was added by me
after measuring the bottleneck, and the equivalence check was my own design.
