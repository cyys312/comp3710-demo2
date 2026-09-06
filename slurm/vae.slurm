#!/bin/bash
#SBATCH --job-name=vae-oasis
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/vae-%j.out
#SBATCH --error=logs/vae-%j.err

# Task 1 — variational autoencoder on the OASIS brain MRI data
# Submit: sbatch slurm/vae.slurm
# Trains two models: latent_dim=2 so the 2D manifold grid can be plotted directly,
#                    latent_dim=32 for sharper reconstructions, with UMAP used to
#                    inspect the structure of the latent space.

# Measured on the cluster (2026-09-02) — three points differ from the usual recipe, and
# copying a template off the web will fail on every one of them:
#   1. Use the a100 partition (AllowAccounts=ALL). On the course-specific comp3710
#      partition every job currently sits in PENDING(PartitionConfig); the account
#      associations have not been set up yet.
#   2. **Do not write --mem**. Every a100 node reports RealMemory=1, so any
#      --mem=<positive value> can never be satisfied and the job pends indefinitely.
#   3. module load cuda is unnecessary: PyTorch 2.13+cu130 in the conda environment
#      torch ships its own CUDA runtime, and the module's cuda/12.2 can clash with it.

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source slurm/_common.sh

python3 recognition/vae_oasis/train.py \
    --data-root "$OASIS_ROOT" \
    --epochs 40 \
    --batch-size 128 \
    --latent-dim "${LATENT_DIM:-32}" \
    --num-workers 4
