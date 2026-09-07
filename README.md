# COMP3710 — Pattern Analysis, Lab Demonstration 2

Semester 2, 2026 · The University of Queensland

Repository: <https://github.com/cyys312/comp3710-demo2>

This repository contains the implementations of the four parts of COMP3710 Lab 2
(Pattern Recognition). Everything is built on **PyTorch**.

---

## Repository layout

```
comp3710/
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
| Part 3.2 — DAWNBench | 4 | **94.15 % / 95 s** (A100) | Done |
| Part 4.1 — edX advanced Git short course | 1 | — | **Outstanding** |
| Part 4 Task 1 — VAE + manifold | — | validation ELBO 4183.05, 2D manifold grid produced | Done |
| Part 4 Task 2 — UNet segmentation | — | test **mean DSC 0.9774**, all four classes > 0.9 | Done |
| Part 4 Task 3 — GAN | — | test diversity **88 %** of real data, no mode collapse | Done |

### Part 1 — runtime of the four DFT implementations (seconds, best of 3 runs)

| Implementation | N=1024 | N=2048 | N=4096 | Complexity |
|---|---|---|---|---|
| NumPy naive DFT (matrix form) | 0.009265 | 0.037120 | 0.146261 | O(N²) |
| PyTorch naive DFT (GPU) | 0.002785 | 0.006693 | 0.010211 | O(N²) |
| NumPy FFT | 0.000006 | 0.000011 | 0.000019 | O(N log N) |
| PyTorch FFT (GPU) | 0.000269 | 0.000236 | 0.000374 | O(N log N) |

Ranking (fast → slow): **NumPy FFT < PyTorch FFT(GPU) < PyTorch naive DFT(GPU) < NumPy naive DFT**.

Three points worth drawing out:

1. **The FFT wins because of the algorithm, not the hardware**. Every doubling of N multiplies the
   naive DFT runtime by roughly 4 (0.0093 → 0.037 → 0.146, exactly O(N²)), while the FFT grows by
   less than 2. No number of parallel cores makes up for a gap in asymptotic complexity.
2. **The GPU buys the naive DFT an order of magnitude but not a better complexity**. At N=4096 the
   GPU version is 14x faster than the CPU one (0.0102 vs 0.1463) because the N² multiply-adds are
   spread across thousands of cores; it is still O(N²), and a few more doublings of N will lose to
   an FFT running on the CPU.
3. **The GPU FFT is in fact slower than NumPy's**, because at this size the fixed cost of each
   kernel launch (about 0.25 ms) already exceeds the computation itself. A GPU only pays off once
   there is enough work to amortise that.

One more note: at the same O(N²), NumPy's matrix form is about 25x faster than the textbook double
Python loop (0.0023 s vs 0.0581 s at N=512) — **complexity and constant factors are two different
things**.

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
