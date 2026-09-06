#!/bin/bash
#SBATCH --job-name=gan-oasis
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/gan-%j.out
#SBATCH --error=logs/gan-%j.err

# Task 3 — GAN generation of OASIS brain MRI
# Submit: sbatch slurm/gan.slurm
#
# The task sheet warns that GAN convergence is erratic and demands that mode collapse be
# solved outright, so this run uses an R1 gradient penalty + generator EMA + TTUR +
# label smoothing, and records the mean pairwise L2 distance between samples as a
# diversity score every epoch.

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source slurm/_common.sh

python3 recognition/gan_oasis/train.py \
    --data-root "$OASIS_ROOT" \
    --epochs "${EPOCHS:-80}" \
    --batch-size 64 \
    --image-size 128 \
    --num-workers 4
