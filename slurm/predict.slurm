#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/predict-%j.out
#SBATCH --error=logs/predict-%j.err

# 在测试集上做推理并生成可视化 —— demo 现场跑的就是这个。
# 提交： sbatch slurm/predict.slurm
# 只跑其中一项： sbatch --export=ALL,ONLY=unet slurm/predict.slurm

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source slurm/_common.sh

ONLY="${ONLY:-all}"

if [ "$ONLY" = "all" ] || [ "$ONLY" = "unet" ]; then
    echo; echo "########## Task 2 UNet —— 测试集分割 ##########"
    python3 recognition/unet_oasis/predict.py \
        --data-root "$OASIS_ROOT" --n-show 4
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "vae" ]; then
    echo; echo "########## Task 1 VAE (latent_dim=2) —— 2D 流形网格 ##########"
    python3 recognition/vae_oasis/predict.py \
        --data-root "$OASIS_ROOT" --mode all \
        --ckpt recognition/vae_oasis/checkpoints/vae_z2.pth

    echo; echo "########## Task 1 VAE (latent_dim=32) —— 重建与隐空间投影 ##########"
    python3 recognition/vae_oasis/predict.py \
        --data-root "$OASIS_ROOT" --mode recon \
        --ckpt recognition/vae_oasis/checkpoints/vae_z32.pth
    python3 recognition/vae_oasis/predict.py \
        --data-root "$OASIS_ROOT" --mode umap \
        --ckpt recognition/vae_oasis/checkpoints/vae_z32.pth
fi

if [ "$ONLY" = "all" ] || [ "$ONLY" = "gan" ]; then
    echo; echo "########## Task 3 GAN —— 生成样本、多样性指标与隐空间插值 ##########"
    python3 recognition/gan_oasis/predict.py \
        --data-root "$OASIS_ROOT" --n 64
fi
