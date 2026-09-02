#!/bin/bash
# Rangpur 作业的公共环境设置，被各个 .slurm 脚本 source。
#
# 集群实测（2026-09-02，sinfo / module avail 核对）：
#   * 分区用 a100（10 台 A100-PCIE-40GB）。课程专用的 comp3710 分区
#     当前对所有人 PENDING(PartitionConfig)，不可用。
#   * 千万别写 --mem：节点 RealMemory=1，带 --mem 的作业永远排不上。
#   * torch 装在 conda 环境 torch 里（2.13.0+cu130），自带 CUDA 运行时，
#     所以不需要 module load cuda —— 加载 12.2 反而可能和自带的 13.0 冲突
#   * OASIS 数据在 /home/groups/comp3710/OASIS（不是 .../keras_png_slices_data）

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

export OASIS_ROOT=/home/groups/comp3710/OASIS
# CIFAR-10 的共享副本（cifar-10-batches-py 格式，torchvision 可直接读）
export CIFAR_ROOT=/home/groups/cifar/CIFAR-10

echo "=============================================="
echo "job      : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-none})"
echo "node     : $(hostname)"
echo "python   : $(which python3)"
python3 -c "import torch; print('torch    :', torch.__version__, '| cuda:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="
