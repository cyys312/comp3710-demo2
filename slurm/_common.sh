#!/bin/bash
# Shared environment setup for Rangpur jobs; sourced by each of the .slurm scripts.
#
# Measured on the cluster (2026-09-02, cross-checked against sinfo / module avail):
#   * Short jobs that must run live go to a100-test first. It has its own two nodes
#     (a100-a, a100-b) rather than sharing the nine a100-0..9, AllowAccounts=ALL, and a
#     10-minute default limit, so nothing can camp on it — measured 2 running / 1 pending
#     while comp3710 had 2 running / 27 pending and the nine shared nodes were held by
#     jobs with 2- and 6-day limits. Keep long training jobs off it for the same reason.
#   * Submit to comp3710,a100 with --account=comp3710. comp3710 sets
#     AllowAccounts=comp3710, but the default account for this login is the personal
#     one, so a job submitted without --account is rejected as PENDING(PartitionConfig).
#     That reason code reads like a misconfigured partition and is really an account
#     mismatch. Naming both partitions lets Slurm pick whichever can start first.
#   * Never write --mem: the nodes report RealMemory=1, so a job carrying --mem is
#     never scheduled.
#   * torch lives in the conda environment named torch (2.13.0+cu130) and ships its
#     own CUDA runtime, so module load cuda is unnecessary -- loading 12.2 can in fact
#     clash with the bundled 13.0
#   * The OASIS data is at /home/groups/comp3710/OASIS (not .../keras_png_slices_data)

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

export OASIS_ROOT=/home/groups/comp3710/OASIS
# Shared copy of CIFAR-10 (cifar-10-batches-py layout, read directly by torchvision)
export CIFAR_ROOT=/home/groups/cifar/CIFAR-10

echo "=============================================="
echo "job      : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-none})"
echo "node     : $(hostname)"
echo "python   : $(which python3)"
python3 -c "import torch; print('torch    :', torch.__version__, '| cuda:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="
