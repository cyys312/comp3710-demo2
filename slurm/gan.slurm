#!/bin/bash
#SBATCH --job-name=gan-oasis
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/gan-%j.out
#SBATCH --error=logs/gan-%j.err

# Task 3 —— OASIS 脑部 MRI 的 GAN 生成
# 提交： sbatch slurm/gan.slurm
#
# 任务书警告 GAN 收敛混乱且要求彻底解决 mode collapse，因此这里用了
# R1 梯度惩罚 + 生成器 EMA + TTUR + label smoothing，
# 并把「样本两两平均 L2 距离」作为多样性指标逐轮记录下来。

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source slurm/_common.sh

python3 recognition/gan_oasis/train.py \
    --data-root "$OASIS_ROOT" \
    --epochs "${EPOCHS:-80}" \
    --batch-size 64 \
    --image-size 128 \
    --num-workers 4
