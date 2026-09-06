# Task 1 — Variational Autoencoder on OASIS Brain MRI

> COMP3710 Lab 2, Part 4 · Easy difficulty (up to 3/7 marks)

## Problem statement

Train a variational autoencoder (VAE) on the Preprocessed OASIS brain MR images and visualise the
manifold formed by its latent space.

## Algorithm

A VAE is made of an encoder `q(z|x)`, reparameterisation sampling and a decoder `p(x|z)`; the
training objective is the negative ELBO:

```
loss = BCE(reconstruction, input) + beta * KL( q(z|x) || N(0, I) )
```

- The **reconstruction term** keeps the decoded output close to the input.
- The **KL term** pulls the posterior towards the standard normal prior, which is what makes the
  latent space continuous and sampleable — and therefore something that can be interpolated and
  visualised as a "manifold" in the first place.
- The reparameterisation trick `z = mu + sigma * eps` keeps the sampling step differentiable.

Network: 4 stride=2 convolutions compress 128×128 down to 8×8, then fully connected layers output
`mu` / `logvar`; the decoder mirrors this with transposed convolutions and ends in a sigmoid over
[0,1].

## Data

| split | directory |
|---|---|
| train | `keras_png_slices_train/` |
| validate | `keras_png_slices_validate/` |
| test | `keras_png_slices_test/` |

The dataset is already split by case and is not re-split, so adjacent slices of the same case
cannot land in the training and test sets at once and leak information.
The VAE is unsupervised: the `keras_png_slices_seg_*` labels take no part in training.

Preprocessing: convert to greyscale, resize to 128×128, normalise pixels to [0,1].

## Running

Locally (Apple MPS / CPU detected automatically):

```bash
python recognition/vae_oasis/train.py --epochs 40 --latent-dim 2    # for the manifold grid
python recognition/vae_oasis/train.py --epochs 40 --latent-dim 32   # sharper reconstructions
python recognition/vae_oasis/predict.py --mode all --ckpt recognition/vae_oasis/checkpoints/vae_z2.pth
```

Rangpur cluster (the data sits at `/home/groups/comp3710/OASIS`, no upload needed):

```bash
sbatch --export=ALL,LATENT_DIM=2  --job-name=vae-z2  slurm/vae.slurm
sbatch --export=ALL,LATENT_DIM=32 --job-name=vae-z32 slurm/vae.slurm
sbatch --export=ALL,ONLY=vae slurm/predict.slurm
```

## Results

Training environment: Rangpur `a100` partition, NVIDIA A100-PCIE-40GB. Each model was trained for
40 epochs at about 6 s/epoch. To get both a directly plottable manifold and sharp reconstructions,
two models were trained:

| latent_dim | best validation ELBO | reconstruction term | KL term | wall time | job |
|---|---|---|---|---|---|
| 2 | 4265.93 | 4173.72 | 8.24 | 4 min 43 s | 581351 |
| 32 | **4183.05** | 4053.01 | 32.46 | 4 min 47 s | 581352 |

(The loss is the negative ELBO per image and the reconstruction term is a BCE summed over 128×128
pixels, which is why the numbers are in the thousands.)

**How to read these two rows**: the 32-dim model reaches a lower ELBO (better reconstruction), at
the price of a latent space too high-dimensional to plot; the 2-dim model reconstructs slightly
worse but lets a grid be laid over the latent plane so the manifold can be drawn directly.
The KL terms behave as expected too: 8.24 nats ≈ 4.1 nats/dim at 2 dimensions, 32.46 nats
≈ 1.0 nats/dim at 32 — the more dimensions, the less information each one carries, but all stay far
from 0, which shows **no posterior collapse occurred** (with KL→0 the latent variable is ignored
and the model degenerates into an ordinary autoencoder).

### Manifold visualisation

`z2_manifold_grid.png` is the central result: a 20×20 grid is laid over the latent plane at the
**quantiles** of the standard normal (not at even spacing) and decoded point by point. Quantiles
are used because the prior is a standard normal — only equal-quantile sampling makes the grid
cover the probability mass evenly, so the edges do not all land in regions never seen in training.

What the figure shows: **neighbouring grid positions transition smoothly**. The butterfly-shaped
central ventricle grows continuously from narrow to wide and from pointed to round along one
direction, with no jumps or tearing. This is exactly the KL term at work — it pulls the posterior
towards the standard normal, forcing the latent space to be "filled in" instead of degenerating
into a set of isolated points, which is why interpolating between two training samples also decodes
into a plausible brain image.

`z32_manifold_umap.png` is a 2D projection of the 32-dim latent space. The cluster's conda
environment has no umap-learn, so the code falls back to PCA automatically (a pure torch SVD
implementation, see `pca_2d`). For the question at hand — is the latent space continuous — a linear
projection is enough; UMAP's strength is preserving non-linear neighbourhood structure, which is
nice to have but changes nothing about the conclusion.

Figures (`outputs/`, prefixed by latent_dim):

- `z*_loss_curve.png` — training / validation loss
- `z*_reconstruction.png` — inputs on the top row, reconstructions below
- `z2_manifold_grid.png` — **2D manifold grid (the visualisation the task sheet asks for)**
- `z*_manifold_umap.png` — 2D projection of the latent space (UMAP, falling back to PCA if absent)
- `z*_recon_epoch{05..40}.png` — how reconstruction quality evolves over training

## Demo notes

- Explain what the KL term does and what happens without it (degenerates into an ordinary
  autoencoder with a discontinuous latent space)
- Explain why the reparameterisation trick is needed
- Lay out the latent_dim trade-off: 2 dimensions can be plotted as a manifold directly but
  reconstruct blurrily, 32 are sharp but need UMAP

## AI usage

See [`AI_PROMPTS.md`](../../AI_PROMPTS.md) in the repository root. The changes relevant to this
sub-project: the scipy dependency was dropped (`normal_ppf` now uses torch's `erfinv`, deviating
from `scipy.stats.norm.ppf` by 8.9e-16), and a PCA fallback was added for UMAP — the cluster
environment has neither, and pulling in a whole dependency for one function is not worth it.
