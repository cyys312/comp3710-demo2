# AI Usage Log

The course marking criteria require that any use of AI tools be explained, including how it was
used and the reasoning behind it, and a prompt history (`prompt_history.pdf` or equivalent) may
be requested. This file records how AI was actually involved while developing this repository,
the **verification and corrections** I applied to its output, and the judgements that remain
mine.

Tool: **Claude Code (Opus 5)**, used interactively on this machine and on the Rangpur cluster.
Date: 2026-09-02.

---

## 1. Where AI Fits In

The marking criteria state that pasting the task straight into an AI and pasting the answer back
is not acceptable use. Here the AI's role was to **speed up implementation and debugging**, while
design decisions, interpretation of results and correctness checks stayed with me. The record
below follows the order "AI output → problem I found → fix"; those fixes are themselves the
evidence of understanding.

---

## 2. Problems I Found and Fixed in the AI's First-Draft Code

### 2.1 Part 1 missed a core requirement of the task sheet

The `dft.py` the AI first wrote only had a NumPy naive DFT compared against `np.fft.fft`.
Re-reading page 5 of the task sheet, the requirement is explicit:

> modify the 'square_wave', 'square_wave_fourier' and 'naive_dft' functions so that
> they are implemented using TensorFlow (TF) or PyTorch operations. For 'naive_dft'
> in particular, create a second version that explicitly runs on the GPU

So a **PyTorch version** is mandatory, and `naive_dft` needs a version that **explicitly runs on
the GPU**. Fix: add `square_wave_torch` / `square_wave_fourier_torch` / `naive_dft_torch`, plus a
timing benchmark that sweeps the problem size. I also added a genuine double-loop version,
`naive_dft_loops`, to show that "the same O(N²) can differ by 25x depending on how it is
written" — the task sheet does not ask for this, but it is the key to understanding *why* the
fastest version is the fastest: **algorithmic complexity and the constant factor are two
different things**.

### 2.2 The DSC used soft Dice, which distorts the reported score

The AI's `dice_per_class` computed Dice straight from the softmax probabilities, with the
training loss and the evaluation metric sharing one function. That is wrong: the DSC the task
sheet asks for is computed on the **hard prediction (argmax)**, i.e. the class map after argmax.
Soft Dice reads low when the model is under-confident and high when it is over-confident, so
reporting it distorts the number.

Fix: split into two functions — `dice_per_class` (soft, differentiable, loss only) and
`hard_dice_counts` (hard, evaluation only).

### 2.3 The DSC aggregation was wrong as well

The original implementation computed one DSC per batch and divided by the number of batches.
DSC is a ratio, so **the mean of per-batch DSCs is not the dataset-level DSC**, and the short
final batch was given the same weight as a full one.

Fix: `hard_dice_counts` returns per-class intersection and cardinality counts; these accumulate
across batches and the division happens once at the end (see the comments in
`recognition/unet_oasis/modules.py`).

### 2.4 The slurm script's parameters were guesses that leave the job unscheduled forever

In the AI's `run_rangpur.slurm`, `--partition=a100`, `module load cuda` and `--mem=32G` were all
guessed from the usual templates. Checking each against the cluster with `sinfo` / `scontrol` /
`sacctmgr` turned up three errors:

| Guess | Measured | Consequence |
|---|---|---|
| `--mem=32G` | every a100 node reports `RealMemory=1` | **job PENDING indefinitely**; the real reason much of the class was stuck at the time |
| `--partition=comp3710` | partition has `AllowAccounts=comp3710`, and the account association was never enabled | always `PENDING (PartitionConfig)`; switched to the `a100` partition |
| `module load cuda` | only cuda/11.1, 11.4 and 12.2 exist, while the conda environment has torch 2.13.0+**cu130** (ships its own runtime) | loading it can actually conflict; the right move is not to load it |

This one shows the AI knows nothing about **a specific cluster's configuration** and can only
offer a template; you have to verify it against the machine yourself.

### 2.5 A multi-process DataLoader is a pessimisation on this machine

The AI defaulted to `num_workers=4/8` plus `persistent_workers=True`. Measured on Apple Silicon:

| Configuration | Time |
|---|---|
| `num_workers=0` | **9 ms/batch** |
| `num_workers=6, persistent=True` | 262 ms/batch |
| `num_workers=6, persistent=False` | 3249 ms/batch |

Decoding one OASIS slice takes about 1 ms, so the IPC serialisation cost of worker processes far
outweighs the benefit — **29x slower**; and fork alongside MPS also left the main process wedged
in an uninterruptible wait. Fix: add `auto_workers()`, which picks by device (4 on CUDA, 0 on
MPS/CPU).

### 2.6 Label decoding allocated an H×W×4 temporary array per image

The original implementation took the distance from every pixel to the 4 reference grey levels
and then an argmin. Replaced by a 256-entry lookup table (`_LABEL_LUT`) that performs the mapping
in a single indexing operation.

---

## 3. Experiments I Added Myself (Not Asked For by the AI)

### 3.1 Part 2: giving the 57.76% figure a frame of reference

The AI stopped once the random forest reached 57.76%. On its own that number means nothing, so I
added two sets of controls:

- **Majority-class baseline**: in LFW, George W Bush alone accounts for 41% of the test set, so
  always guessing him already scores 41.3%.
- **Ablation on the number of principal components**: 25→0.6304, **50→0.6553**, 100→0.5807,
  150→0.5776. More components make it **worse**, not better. Explanation: the higher-order
  eigenfaces mostly encode lighting and pose noise; with the random forest's default
  `max_features='sqrt'`, at 150 dimensions each split samples only about 12 features, so it is
  more likely to draw noise.
- **Class weighting**: `class_weight='balanced'` lifts accuracy from 0.5776 to 0.7298 and macro-F1
  from 0.3022 to 0.6361 — the confusion matrix shows the original model assigned nearly every
  sample to the majority class.

### 3.2 Part 3.2: measure first, then optimise

The DAWNBench plan the AI gave used torchvision's CPU DataLoader. Measured on an A100,
**227 seconds was not even enough for one epoch** (97 steps, >2.3 s/step), while the actual
compute for ResNet-18 on an A100 takes tens of milliseconds — the bottleneck is the data
pipeline, not the model.

Cause: each a100 node has only 8 CPU cores, and PIL's
`RandomCrop → Flip → ToTensor → Normalize` cannot keep the GPU fed.

What I did: make all of CIFAR-10 GPU-resident (153 MB as uint8) in the 40 GB of device memory,
and perform the random crop and flip with tensor ops on the GPU (`dataset.GPUCifar`). Result:
**1.9 s/epoch**, and **95 seconds to reach 94.15%** over 30 epochs — nearly 4x faster than the
task sheet's V100 baseline (360 seconds).

I verified the equivalence: after reflect padding to 40×40, a per-sample random offset in `[0,8]`
crops back to 32×32, and when the offset is exactly (4,4) the original image is **reproduced
exactly** — which confirms the indexing is not off by one.

---

## 4. Judgements That Stayed With Me and the AI Could Not Make

- Choosing between the task sheet's three difficulty tiers (Easy/Medium/Hard) and the mark ceiling
  that follows from it
- Keeping the dataset's own train/validate/test split (separated by patient) rather than
  re-splitting — adjacent slices are highly correlated, and a random split would put slices from
  the same patient into both train and test, causing leakage
- Label resizing must use nearest-neighbour rather than bilinear, otherwise interpolation invents
  intermediate values outside `{0,85,170,255}`
- The reason the UNet loss is half CE and half Dice (background is 72.29% of the pixels, so pure
  CE biases towards it)
- Obtaining and interpreting every measured number above

---

## 5. Evidence Available

- This file (the development record and the log of corrections)
- The git commit history (grouped by Part; each commit message states why the change was made)
- The result figures under each `outputs/` directory and the full training logs under `logs/`
- Cluster job IDs: DAWNBench `581349`, UNet `581337`, VAE `581351`/`581352`
