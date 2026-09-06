# Task 3 — GAN Generation of OASIS Brain MRI

> COMP3710 Lab 2, Part 4 · Hard difficulty (all three tasks attempted, capped at 7/7 marks)

## Problem statement

Generate Preprocessed OASIS brain MR images with a generative adversarial network. The task sheet
requires the images to be convincingly realistic, requires evidence of training (generated
samples, loss curves and so on), and states that **problems such as mode collapse must be fully
resolved**.

## Results

Rangpur `a100` partition, NVIDIA A100-PCIE-40GB, job `581557`:
128×128 resolution, 80 epochs, about 6 s/epoch, **9.0 minutes in total**.

| Metric | End of training | Independent check (`predict.py`) |
|---|---|---|
| Diversity score of generated samples (mean pairwise L2 distance) | 26.20 | 26.70 |
| Same score on real samples (baseline) | 31.24 | 30.31 |
| **Ratio** | **84 %** | **88 %** |
| Verdict | no significant mode collapse | no significant mode collapse |

The two columns differ in **where the real-data baseline comes from**: during training it is the
last training batch of that epoch, whereas `predict.py` uses the **test set** (544 images that
never took part in training). The latter is the cleaner control; it agrees with the training
figure and is slightly better, which shows the score itself is stable rather than a cherry-picked
number.

Diversity across training (`outputs/diversity_curve.png`):

| epoch | 6 | 10 | 15 | 45 | 75 | 80 |
|---|---|---|---|---|---|---|
| fraction of the real-data score | 24 % | 44 % | 65 % | 84–90 % | 88 % | 84 % |

**This V-shaped curve is worth explaining**: early in training the generator first learns the
"average brain" — every sample looks almost the same, so diversity falls all the way to 24 %.
Only then does it start learning variation, and the score recovers and settles around 85 %.
**Falling before rising is normal; never recovering is mode collapse.**

## Why not a naive DCGAN

The task sheet explicitly warns that GAN convergence is messy. Five targeted measures are used
here, each aimed at one specific failure mode:

| Measure | Failure mode it targets |
|---|---|
| **R1 gradient penalty** (on real samples only, every 16 steps) | Discriminator too strong → vanishing generator gradients. R1 constrains the discriminator to be locally Lipschitz; unlike WGAN-GP it needs no interpolation between real and fake samples, so it costs less compute |
| **EMA of the generator weights** | Single-step weight oscillation makes sample quality swing. Sampling from the moving-average weights gives visibly steadier images |
| **TTUR** (D learning rate 3e-4 > G's 2e-4) | The two networks updating at mismatched speeds |
| **Real labels smoothed to 0.9** | The discriminator driving its confidence into saturation, where gradients approach 0 |
| **InstanceNorm instead of BatchNorm in the discriminator** | BatchNorm couples samples within a batch to each other, a known source of instability in GANs |

The loss also uses the **non-saturating form** (maximise `log D(G(z))` rather than minimise
`log(1-D(G(z)))`) — the latter's gradient is almost 0 early in training while the discriminator is
strong, a problem already pointed out in the original GAN paper.

## How mode collapse is ruled out

"Do the images look good" is subjective, so this project rests on two **objective** lines of
evidence:

1. **Diversity score** (`modules.diversity_score`): draw random samples, pair them up and take
   the mean L2 distance, then form the ratio against the same score on real data. A collapsed
   generator produces near-identical samples, so this number falls far below the real data.
   Measured at 84 %, and recorded epoch by epoch as a curve.
2. **Latent-space interpolation** (`outputs/interpolation.png`): interpolate linearly between two
   latent vectors and decode point by point, 8 rows of 10 steps each. A collapsed generator
   **jumps** part way along (hard-switching from one mode to another); a healthy one should
   transition smoothly.
   **What actually happens**: every row is a continuous deformation — the ventricles go from wide
   to narrow, contracting from a butterfly shape into a slit, while the skull outline adjusts in
   step, with no jumps anywhere. This corroborates the 88 % diversity score.

## Known limitations (raise these during the demo)

- **The samples are too left-right symmetric.** Real brain slices are approximately, not strictly,
  symmetric, and some of the generated samples are symmetric to an implausibly neat degree. The
  likely cause is that the data itself has been registered and that 128×128 leaves little fine
  asymmetric detail, so the generator learned "symmetry" as a shortcut.
- **Only mid-axial slices are covered.** This faithfully reflects the training data — Preprocessed
  OASIS slices are themselves concentrated in that range — rather than being a flaw of the model.
- **No standard metric such as FID.** The diversity score can establish "no collapse" but cannot
  fully measure "how realistic"; in this course the latter is judged subjectively by the
  demonstrator.

## Data

Uses the training split shipped with the dataset (9664 images). There is one crucial difference
from the VAE loader: **pixels are normalised to [-1, 1] rather than [0, 1]**, because the last
layer of the generator is a tanh. Real and fake data must share the same value range, otherwise
the discriminator can tell them apart from the range alone and training degenerates immediately.

The GAN is unsupervised, so the `keras_png_slices_seg_*` labels take no part in training.
A GAN has no validation likelihood to early-stop on, so no validation split is made.

## Running

```bash
# Cluster (recommended, finishes in 9 minutes)
sbatch slurm/gan.slurm
sbatch --export=ALL,ONLY=gan slurm/predict.slurm

# Local
python recognition/gan_oasis/train.py --epochs 80
python recognition/gan_oasis/predict.py --n 64
```

## Figures (`outputs/`)

- `samples_epoch{005..080}.png` — fixed-noise samples across training (evidence of training)
- `loss_curve.png` — discriminator/generator losses
- `diversity_curve.png` — **diversity score vs the real-data baseline (quantitative evidence on
  mode collapse)**
- `generated_samples.png` — final grid of generated brain images
- `interpolation.png` — latent-space interpolation

## Demo notes

- Explain the GAN minimax game, and why it is harder to train than a VAE
  (there is no single objective to optimise; two networks chase each other after a moving target)
- Explain why the non-saturating loss is used instead of the form in the original paper
- **Focus on how mode collapse is judged**: not by how good the images look, but by the 84 %
  diversity score and the smooth transitions in the interpolation figure; and explain why the
  V-shaped curve is normal
- Explain what the R1 penalty does, and why it is applied "lazily" every 16 steps rather than
  every step
- Contrast with the VAE of Task 1: the VAE has an explicit likelihood and trains stably but
  generates blurrier images; the GAN has no explicit likelihood and trains less stably, but its
  detail is sharper

## AI usage

See [`AI_PROMPTS.md`](../../AI_PROMPTS.md) in the repository root.
