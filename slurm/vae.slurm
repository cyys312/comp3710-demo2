#!/bin/bash
#SBATCH --job-name=vae-oasis
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/vae-%j.out
#SBATCH --error=logs/vae-%j.err

# Task 1 —— OASIS 脑部 MRI 的变分自编码器
# 提交： sbatch slurm/vae.slurm
# 训练两个模型：latent_dim=2 用于直接画 2D 流形网格，
#              latent_dim=32 重建更清晰，用 UMAP 看隐空间结构。

# 集群实测（2026-09-02）——三处和常见写法不同，照抄网上的模板会挂：
#   1. 分区用 a100（AllowAccounts=ALL）。课程专用的 comp3710 分区目前
#      所有人的作业都卡在 PENDING(PartitionConfig)，账号关联还没开通。
#   2. **不要写 --mem**。所有 a100 节点的 RealMemory=1，
#      任何 --mem=<正数> 都永远满足不了，作业会无限期 pending。
#   3. 不需要 module load cuda：conda 环境 torch 里的 PyTorch 2.13+cu130
#      自带 CUDA 运行时，加载 module 的 cuda/12.2 反而可能冲突。

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source slurm/_common.sh

python3 recognition/vae_oasis/train.py \
    --data-root "$OASIS_ROOT" \
    --epochs 40 \
    --batch-size 128 \
    --latent-dim "${LATENT_DIM:-32}" \
    --num-workers 4
