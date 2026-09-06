# Task 2 — UNet Segmentation of OASIS Brain MRI

> COMP3710 Lab 2, Part 4 · Medium difficulty (up to 5/7 marks combined with Task 1)

## Problem statement

Segment the Preprocessed OASIS brain MR images with a UNet.
**Every label must reach a DSC (Dice Similarity Coefficient) > 0.9**, and the output has to be in
one-hot / categorical form.

## Algorithm

UNet is an encoder-decoder architecture whose core idea is the **skip connection**:
the high-resolution features of each encoder level are concatenated into the matching
decoder level.
Downsampling lets the network see a wide context (which tissue this is), and the skip connections
bring back the spatial detail that was lost (where the boundary is); only together do they give a
segmentation that is both semantically correct and sharp at the edges.

- Encoder: 4 levels of `DoubleConv` + MaxPool, channels 32→64→128→256, bottleneck 512
- Decoder: transposed-convolution upsampling + concat with the encoder features + `DoubleConv`
- Output head: 1×1 convolution → 4-channel logits, paired with one-hot labels

Loss = 0.5 × CrossEntropy + 0.5 × Dice Loss.
Pure cross-entropy leans towards the background under class imbalance (the background dominates);
Dice Loss optimises the evaluation metric itself.

## Data

| split | images | labels |
|---|---|---|
| train | `keras_png_slices_train/` | `keras_png_slices_seg_train/` |
| validate | `keras_png_slices_validate/` | `keras_png_slices_seg_validate/` |
| test | `keras_png_slices_test/` | `keras_png_slices_seg_test/` |

The split shipped with the dataset is kept (separated by case, so adjacent slices cannot leak).
Images and labels are paired by the filename index: `case_441_slice_0` ↔ `seg_441_slice_0`.

Preprocessing: resize to 256×256 (**labels must use nearest-neighbour interpolation**, or the
interpolation invents intermediate class values that do not exist); the label greyscales
{0, 85, 170, 255} are mapped to class indices {0,1,2,3} and converted to one-hot for training.

## Running

Locally (Apple MPS / CPU detected automatically):

```bash
python recognition/unet_oasis/train.py --epochs 25 --batch-size 16
python recognition/unet_oasis/predict.py --n-show 4      # run this one live in the demo
```

Rangpur cluster (the data sits at `/home/groups/comp3710/OASIS`, no upload needed):

```bash
sbatch slurm/unet.slurm
sbatch --export=ALL,ONLY=unet slurm/predict.slurm
```

## Results

Training environment: Rangpur `a100` partition, NVIDIA A100-PCIE-40GB (job 581337).
25 epochs at roughly 0.4 min/epoch, about 10 minutes in total.
The table below is measured on the **544 test slices**, a split that took no part in training or in
any tuning:

| class | tissue | pixel share | test DSC |
|---|---|---|---|
| 0 | background | 72.30 % | **0.9993** |
| 1 | CSF | 5.66 % | **0.9651** |
| 2 | grey matter | 11.39 % | **0.9658** |
| 3 | white matter | 10.65 % | **0.9795** |
| **mean** | | | **0.9774** |

**All four labels are above 0.9, which satisfies the task sheet.** The best checkpoint comes from
epoch 22 (chosen by mean validation DSC; the test set was evaluated only once, at the very end).

### How the label-to-tissue mapping was established

The dataset ships no label dictionary. I recovered it from the mean greyscale of each label region
in the corresponding image:

| label | mean greyscale |
|---|---|
| c0 | 0.0551 |
| c1 | 0.1407 |
| c2 | 0.3153 |
| c3 | 0.4755 |

The greyscales increase strictly monotonically, matching the tissue contrast order of T1-weighted
MRI exactly: background darkest → CSF (hypointense on T1) → grey matter → white matter brightest.

### Why c1 has the lowest DSC

CSF covers only 5.66 % of the pixels and is mostly thin structures along the ventricle margins, so
boundary pixels make up a larger fraction of its area than of any other class — Dice is most
sensitive to boundary error on small targets, and being one ring of pixels off costs a small
structure far more than a large one. This is also why the loss combines CE + Dice rather than CE
alone: the background is 72.30 % of the pixels, and pure CE would bias the model towards it.

Figures (`outputs/`):

- `dice_curve.png` — per-class DSC against epoch (with the 0.9 reference line)
- `segmentation_examples.png` — three columns: image / ground truth / prediction

## Demo notes

- Run `predict.py` on the test set live to show the inference results (the task sheet requires it)
- Explain why the skip connection is decisive for segmentation accuracy
- Explain where the per-class DSC differences come from (small structures are harder, boundary
  pixels are a larger share of their area)
- Explain why labels must be resized with nearest-neighbour interpolation

## AI usage

See [`AI_PROMPTS.md`](../../AI_PROMPTS.md) in the repository root. Two items bear directly on this
sub-project: the first AI draft used soft Dice as both the loss and the evaluation metric
(evaluation has to use the hard prediction (argmax)), and it averaged batch-level DSC into a
dataset-level DSC (a ratio cannot be averaged that way). Both are fixed, with the reasoning written
into the comments in `modules.py`.
