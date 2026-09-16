#!/bin/bash
#SBATCH --job-name=demo-dawnbench
#SBATCH --partition=comp3710,a100
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/demo-dawnbench-%j.out
#SBATCH --error=logs/demo-dawnbench-%j.err

# Part 3.2 demonstration script — the task sheet requires:
#   "The model must run inference and a single epoch of training on the
#    Ranpur compute cluster during the demonstration." (1 Mark)
#
# This script covers both in one go: inference from a trained checkpoint, then training.
#
# The sheet only asks for a single epoch, but the full 30 take about 95 s, so the demo runs
# all of them. That turns one run into evidence for all three of the Part 3.2 criteria:
# the >90 % tier, the live-run tier, and the 94 %-under-360 s tier are all decided by
# numbers produced in front of the demonstrator rather than quoted from the README.
# train.py only prints the [1 mark] / [2 marks] verdict lines when epochs >= 10.
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
echo "########## 2/2  Training: 30 epochs from scratch, about 95 s ##########"
python3 part3_cnn/dawnbench/train.py \
    --data-root "$CIFAR_ROOT" \
    --epochs 30 --batch-size 512 --no-save
# --no-save still matters even now that the run is a full one. Saving triggers on
# acc > best_acc with best_acc starting from 0, so a demo run that happened to land
# slightly below the stored checkpoint would still replace it — and the inference step
# above depends on that file. The demo should never be able to damage its own evidence.
