#!/bin/bash
#SBATCH --job-name=demo-dawnbench
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/demo-dawnbench-%j.out
#SBATCH --error=logs/demo-dawnbench-%j.err

# Part 3.2 demonstration script — the task sheet requires:
#   "The model must run inference and a single epoch of training on the
#    Ranpur compute cluster during the demonstration." (1 Mark)
#
# This script covers both in one go: inference from a trained checkpoint, then a single
# epoch of training.
# Submit: sbatch slurm/demo_dawnbench.slurm

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source slurm/_common.sh

echo
echo "########## 1/2  Inference: load the trained checkpoint, evaluate on the test set ##########"
python3 part3_cnn/dawnbench/predict.py \
    --data-root "$CIFAR_ROOT" \
    --ckpt part3_cnn/dawnbench/checkpoints/resnet18_cifar10.pth

echo
echo "########## 2/2  Training: one epoch from scratch to show the pipeline runs ##########"
python3 part3_cnn/dawnbench/train.py \
    --data-root "$CIFAR_ROOT" \
    --epochs 1 --batch-size 512 --no-save
# --no-save matters: without it the model from this one epoch overwrites the 94.15%
# checkpoint the inference step above depends on (saving triggers on acc > best_acc,
# and best_acc starts from 0)
