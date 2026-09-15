#!/bin/bash
#SBATCH --job-name=unet-oasis
#SBATCH --partition=comp3710,a100
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/unet-%j.out
#SBATCH --error=logs/unet-%j.err

# Task 2 — UNet segmentation of the OASIS brain MRI data
# Submit: sbatch slurm/unet.slurm
# Target: DSC > 0.9 on all 4 labels

# Measured on the cluster (2026-09-02) — three points differ from the usual recipe, and
# copying a template off the web will fail on every one of them:
#   1. Submit to comp3710,a100 with --account=comp3710. The course partition sets
#      AllowAccounts=comp3710 while the default account for this login is the personal
#      one (s4984244), so without --account every job is rejected with
#      PENDING(PartitionConfig) — which reads like a broken partition but is not.
#      Listing both partitions lets Slurm start the job wherever a GPU frees up first;
#      a100 is often fully held by multi-day jobs, and comp3710 usually is not.
#   2. **Do not write --mem**. Every a100 node reports RealMemory=1, so any
#      --mem=<positive value> can never be satisfied and the job pends indefinitely.
#   3. module load cuda is unnecessary: PyTorch 2.13+cu130 in the conda environment
#      torch ships its own CUDA runtime, and the module's cuda/12.2 can clash with it.

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source slurm/_common.sh

python3 recognition/unet_oasis/train.py \
    --data-root "$OASIS_ROOT" \
    --epochs 25 \
    --batch-size 32 \
    --image-size 256 \
    --num-workers 4
