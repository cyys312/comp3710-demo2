#!/bin/bash
# Shared environment setup for Rangpur jobs; sourced by each of the .slurm scripts.
#
# Measured on the cluster (2026-09-02, cross-checked against sinfo / module avail):
#   * Use the a100 partition (10 x A100-PCIE-40GB). The course-specific comp3710
#     partition is stuck in PENDING(PartitionConfig) for everyone and is unusable.
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
