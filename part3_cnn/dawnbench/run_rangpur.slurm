#!/bin/bash
#SBATCH --job-name=dawnbench
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/dawnbench-%j.out
#SBATCH --error=logs/dawnbench-%j.err

# Part 3.2 DAWNBench — ResNet-18 / CIFAR-10
# Submit: sbatch part3_cnn/dawnbench/run_rangpur.slurm
# Target: >94% accuracy, training time matching or beating the 360 s V100 reference.

# Measured on the cluster (2026-09-02) — these points differ from the usual recipe, and
# copying a template off the web will fail on every one of them:
#   1. Use the a100 partition (AllowAccounts=ALL). On the course-specific comp3710
#      partition every job currently sits in PENDING(PartitionConfig); the account
#      associations have not been set up yet.
#   2. **Do not write --mem**. Every a100 node reports RealMemory=1, so any
#      --mem=<positive value> can never be satisfied and the job pends indefinitely.
#   3. A shared copy of CIFAR-10 already exists on the cluster: /home/groups/cifar/CIFAR-10
#      (torchvision's cifar-10-batches-py layout, so just point straight at it).
#      Never download it on the spot: torchvision's default mirror cs.toronto.edu only
#      reaches the cluster at ~3 KB/s, so 170 MB takes upwards of ten hours; the cluster
#      pulls 2.5 MB/s from GitHub, so the mirror is at fault, not the link.
#   4. module load cuda is unnecessary: PyTorch 2.13+cu130 in the conda environment
#      torch ships its own CUDA runtime, and the module's cuda/12.2 can clash with it.

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source slurm/_common.sh

python3 part3_cnn/dawnbench/train.py \
    --epochs 30 \
    --batch-size 512 \
    --num-workers 4 \
    --data-root "$CIFAR_ROOT"
