# COMP3710 — Pattern Analysis, Lab Demonstration 2

Semester 2, 2026 · The University of Queensland

Repository: <https://github.com/cyys312/comp3710-demo2>

This repository contains the implementations of the four parts of COMP3710 Lab 2
(Pattern Recognition). Everything is built on **PyTorch**.

---

## Repository layout

```
comp3710-demo2/
├── data/                       # Datasets (not in git), see "Data" below
├── docs/                       # Task sheet and marking rubric PDFs
├── part1_dft/                  # Part 1 — Discrete Fourier Transform (1 mark)
│   └── dft.py
├── part2_eigenfaces/           # Part 2 — Eigenfaces / PCA + random forest (1 mark)
│   └── eigenfaces.py
├── part3_cnn/                  # Part 3 — CNN (5 marks)
│   ├── cnn_lfw.py              #   3.1 LFW face CNN classifier (1 mark)
│   ├── ablation_mlp.py         #       control: is the convolution needed?
│   └── dawnbench/              #   3.2 DAWNBench: ResNet-18 + CIFAR-10 (4 marks)
│       ├── modules.py          #       Model definitions
│       ├── dataset.py          #       Data loading and augmentation
│       ├── train.py            #       Training (mixed precision + OneCycle)
│       ├── predict.py          #       Inference and evaluation
│       └── run_rangpur.slurm   #       Rangpur cluster submission script
├── recognition/                # Part 4 — Recognition tasks (7 marks)
│   ├── vae_oasis/              #   Task 1 — VAE (Easy, 3 marks)
│   ├── unet_oasis/             #   Task 2 — UNet segmentation (Medium, 5 marks total)
│   └── gan_oasis/              #   Task 3 — GAN brain synthesis (Hard, 7 marks total)
├── slurm/                      # Rangpur job scripts (settings verified on the cluster)
├── AI_PROMPTS.md               # AI usage log (required by the rubric)
├── DEMO_PREP.md                # Demo prep: expected questions and how to answer them
├── CODE_WALKTHROUGH.md         # Code walkthrough: what each file does, block by block
├── docs/demo_handbook.html     # The two above merged, ready to open in a browser
├── requirements.txt
└── README.md
```

Every `recognition/` sub-project follows the four-file structure the course requires:

| File | Responsibility |
|---|---|
| `modules.py` | Definitions of the model / network components |
| `dataset.py` | Data loaders and preprocessing |
| `train.py` | Training, validation, checkpointing, loss curves |
| `predict.py` | Loads a checkpoint for inference and visualisation |
| `README.md` | Algorithm background, data splits, results and figures |

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For a GPU build of PyTorch, use the command for your CUDA version from <https://pytorch.org>.

## Data

- **LFW (Parts 2 / 3.1)**: downloaded automatically by `sklearn.datasets.fetch_lfw_people`
  on the first run.
- **CIFAR-10 (Part 3.2)**: locally, `torchvision.datasets.CIFAR10` downloads it into `data/`.
  **Do not download it on the cluster** — a shared copy already exists at
  `/home/groups/cifar/CIFAR-10` (`cifar-10-batches-py` format, which torchvision reads directly);
  the default torchvision mirror only reaches about 3 KB/s from the cluster, so 170 MB would take
  more than ten hours.
- **OASIS brain MRI (Part 4)**: unzip locally into `data/keras_png_slices_data/`; on the cluster
  it already lives at `/home/groups/comp3710/OASIS`, so nothing needs uploading.

```bash
unzip keras_png_slices_data.zip -d data/
```

The directory holds six subsets: `keras_png_slices_{train,validate,test}` (raw images)
and `keras_png_slices_seg_{train,validate,test}` (segmentation labels).

## Running

```bash
python part1_dft/dft.py
python part2_eigenfaces/eigenfaces.py
python part3_cnn/cnn_lfw.py
python part3_cnn/ablation_mlp.py            # CNN vs plain fully connected
python part3_cnn/dawnbench/train.py --epochs 30
python recognition/vae_oasis/train.py
python recognition/unet_oasis/train.py
```

## Training on Rangpur

Part 3.2 (DAWNBench) has to run on the cluster, and the Part 4 VAE/UNet are far faster there too
(one UNet epoch: about 0.4 minutes on an A100 vs about 7 minutes on an Apple M5).

```bash
# 1) First checkout (off campus, connect the UQ VPN first or ssh to port 22 will time out)
ssh <uqusername>@rangpur.compute.eait.uq.edu.au
git clone https://github.com/cyys312/comp3710-demo2.git
cd comp3710-demo2

# 2) Submit the jobs. Every dataset is already on the cluster; nothing to upload or download.
sbatch part3_cnn/dawnbench/run_rangpur.slurm     # Part 3.2
sbatch slurm/unet.slurm                          # Task 2
sbatch --export=ALL,LATENT_DIM=2 --job-name=vae-z2 slurm/vae.slurm   # Task 1
sbatch slurm/predict.slurm                       # inference and visualisation (run this live)

squeue -u $USER
tail -f logs/<jobname>-<jobid>.out
```

### Three cluster-configuration traps (verified on the cluster; copied templates will fail)

Cross-checking with `sinfo` / `scontrol show node` / `sacctmgr` shows that the slurm templates
commonly found online fail outright on Rangpur:

| Common recipe | What is actually true | Consequence |
|---|---|---|
| `#SBATCH --mem=32G` | all a100 nodes report `RealMemory=1` | **job PENDING forever** — the real reason many people get stuck |
| `--partition=comp3710` | partition sets `AllowAccounts=comp3710`; account association not enabled | always `PENDING (PartitionConfig)`; use the `a100` partition |
| `module load cuda` | only cuda/11.1, 11.4, 12.2 exist; the conda env has torch 2.13.0+**cu130**, which ships its own runtime | version clash risk; the right move is not to load it |

Also, **never run any compute on the login node (login0)**, downloading datasets included — it is
shared by the whole course. If something has to be downloaded, submit a `--partition=cpu` job.
All the details are written up in the comments of `slurm/_common.sh`.

## Results overview

Every number here was measured; the commands and logs are in the per-directory READMEs.

| Part | Marks | Result | Status |
|---|---|---|---|
| Part 1 — DFT | 1 | timing comparison of four implementations, size sweep (table below) | Done |
| Part 2 — Eigenfaces + random forest | 1 | accuracy 0.5776 (baseline 0.4130), 0.7298 weighted | Done |
| Part 3.1 — LFW CNN | 1 | accuracy **0.8509**, clearly ahead of Part 2 | Done |
| Part 3.2 — DAWNBench | 4 | **94.15 % / 95 s** (A100), reproduced at 94.08 % / 74 s | Done |
| Part 4.1 — edX advanced Git short course | 1 | — | **Outstanding** |
| Part 4 Task 1 — VAE + manifold | — | validation ELBO 4183.05, 2D manifold grid produced | Done |
| Part 4 Task 2 — UNet segmentation | — | test **mean DSC 0.9774**, all four classes > 0.9 | Done |
| Part 4 Task 3 — GAN | — | test diversity **88 %** of real data, no mode collapse | Done |

### Part 1 — runtime of the four DFT implementations

Measured on Apple MPS (local machine), best of 3 runs per size. The raw console output of this
exact run is kept at [`part1_dft/outputs/part1_run.log`](part1_dft/outputs/part1_run.log), and
[`part1_dft/outputs/dft_timing.png`](part1_dft/outputs/dft_timing.png) plots the same data —
the table, the figure and the log all come from one invocation of
`python part1_dft/dft.py --sizes 256 512 1024 2048 4096`.

| Implementation | N=256 | N=512 | N=1024 | N=2048 | N=4096 | Complexity |
|---|---|---|---|---|---|---|
| NumPy naive DFT (matrix form) | 0.000572 | 0.002238 | 0.009335 | 0.037196 | 0.145681 | O(N²) |
| PyTorch naive DFT (MPS) | 0.000587 | 0.000835 | 0.001576 | 0.006524 | 0.010210 | O(N²) |
| NumPy FFT | 0.000003 | 0.000004 | 0.000006 | 0.000011 | 0.000018 | O(N log N) |
| PyTorch FFT (MPS) | 0.000418 | 0.000206 | 0.000212 | 0.000224 | 0.000175 | O(N log N) |

Ranking at N=4096 (fast → slow): **NumPy FFT < PyTorch FFT(MPS) < PyTorch naive DFT(MPS) <
NumPy naive DFT**. The script prints this ranking line for every N, because the order is not the
same at every size — see point 4.

Four points worth drawing out:

1. **The FFT wins because of the algorithm, not the hardware.** Every doubling of N multiplies the
   naive DFT runtime by almost exactly 4 (0.009335 → 0.037196 → 0.145681, ratios 3.99 and 3.92 —
   textbook O(N²)), while the FFT grows by less than 2. No number of parallel cores makes up for a
   gap in asymptotic complexity.
2. **The GPU buys the naive DFT an order of magnitude but not a better complexity.** At N=4096 the
   GPU version is 14x faster than the CPU one (0.010210 vs 0.145681) because the N² multiply-adds
   are spread across many cores; it is still O(N²), and a few more doublings of N will lose to an
   FFT running on the CPU.
3. **The GPU FFT is in fact slower than NumPy's** at every size measured, because the fixed cost of
   a kernel launch (a floor of about 0.2 ms, visible in the almost flat N=512..4096 row) already
   exceeds the computation itself. A GPU only pays off once there is enough work to amortise that.
4. **At small N the ranking inverts.** At N=256 the naive DFT *on the GPU* is the slowest of all
   four (0.59 ms, behind even the CPU version at 0.57 ms): the transfer and launch overhead costs
   more than the whole 256² of arithmetic it was meant to accelerate. The GPU only overtakes the
   CPU from N=512 on. The asymptotics take over only once N is large enough for the work to
   dominate — which is exactly what the task sheet's "change the size of the data" step exposes.

A caveat worth stating at the demo: these are laptop measurements, and individual entries move by
10–30 % between runs. What is stable, and what the argument rests on, is the *ordering* and the
*slopes* — not the third decimal place of any one number.

One more note: at the same O(N²), NumPy's matrix form is about 24x faster than the textbook double
Python loop (0.002400 s vs 0.0585 s at N=512) — **complexity and constant factors are two different
things**.

### Part 1 — DFT components vs the coefficients used to build the wave

`square_wave_fourier` gives harmonic n the coefficient 4/(nπ), so the DFT can be checked against
the values actually used to synthesise the signal rather than against another FFT implementation:

| harmonic n | measured | 4/(nπ) | rel. error |
|---|---|---|---|
| 1 | 1.273240 | 1.273240 | 0.00e+00 |
| 3 | 0.424413 | 0.424413 | 1.31e-16 |
| 5 | 0.254648 | 0.254648 | 4.36e-16 |
| 7 | 0.181891 | 0.181891 | 4.58e-16 |
| 9 | 0.141471 | 0.141471 | 5.89e-16 |
| 11 | 0.115749 | 0.115749 | 1.32e-15 |

They agree to machine precision, and the two *empty* places in the spectrum are the interesting part:

- **even bin n=2 → 5.7e-18.** A square wave is half-wave symmetric, so the even harmonics cancel.
- **first missing odd bin n=101 → 8.8e-17.** Only 50 harmonics were synthesised, so there is nothing
  above n=99. A true square wave would carry energy there for ever — this is precisely the
  difference between the reconstruction and the thing it approximates.

Agreement is this exact only because T=1 s and f0=1 Hz place every harmonic on a bin centre, so
there is no spectral leakage. Shift f0 off an integer and each line smears into its neighbours.

### Part 3.2 — the full run, reproduced

Both figures come from real runs on the cluster, not from a single lucky one:

| Run | Accuracy | Time | `[1 mark]` | `[2 marks]` |
|---|---|---|---|---|
| Job 581349 (original) | 94.15 % | 95 s | met | met |
| Job 592350 (re-run) | 94.08 % | 74 s | met | met |

The console output of the second is kept verbatim at
[`part3_cnn/dawnbench/outputs/full_run_592350.log`](part3_cnn/dawnbench/outputs/full_run_592350.log),
including the header that records the node, the torch build and the GPU, and the two verdict
lines `train.py` prints against the task sheet's tiers.

The ~0.07 point and ~20 second spread between the two runs is ordinary variation — different
random initialisation, and a differently loaded node. **What matters is that both clear both
thresholds**, so the result is a property of the method rather than of one run.

### Outstanding

- [ ] **Part 4.1 — edX advanced Git short course** (1 mark, purely time, no code to write)

Tasks 1, 2 and 3 are all implemented, so Part 4 counts at **Hard difficulty** and the task
component caps out at 7/7.

### On mode collapse in Task 3

The task sheet requires that mode collapse «need to be fully resolved». This project does not rely
on eyeballing the samples: it uses the **mean pairwise L2 distance between samples** as the
diversity score, and reports it as a ratio against the same measure on real data:

| epoch | 6 | 15 | 43 | 80 (final) |
|---|---|---|---|---|
| Diversity / real data | 24 % | 65 % | 90 % | **84 %** |

`predict.py` re-checks this independently against the test set (544 images, never seen in training)
and gets **88 %**, the same conclusion.

The V-shaped curve is normal early GAN behaviour — the generator first learns an "average brain"
(brain-like, but all alike), and only once the discriminator sees through it is it forced to cover
the real distribution. **A dip is not a collapse; the difference is whether it recovers by itself**.
See [`recognition/gan_oasis/README.md`](recognition/gan_oasis/README.md) for details.

## AI usage declaration

Parts of the code in this repository were written with the help of **Claude Code (Opus 5)**, in
line with the task sheet's "Use of Artificial Intelligence" section and the rubric's
"Fair AI Usage" requirement.

The full log is in [`AI_PROMPTS.md`](AI_PROMPTS.md), which lists the **6 problems I found and
fixed** in the AI's first-draft code (including two bugs that would have distorted the reported DSC
and three slurm settings that would have left cluster jobs queued forever), along with the control
experiments I ran myself.
