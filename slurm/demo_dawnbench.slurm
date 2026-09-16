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
# EPOCHS defaults to 1, which is what the sheet asks for and what the demonstrator asked
# for. Submitting with EPOCHS=30 instead runs the real thing in about 95 s and makes
# train.py print its [1 mark] / [2 marks] verdict lines, which it suppresses below ten
# epochs so that a pipeline check cannot read as a failed task. Use that when there is
# time, because it decides all three Part 3.2 criteria from numbers computed during the
# demonstration rather than quoted from the README.
#
# Submit: sbatch slurm/demo_dawnbench.slurm
#         sbatch --export=ALL,EPOCHS=30 slurm/demo_dawnbench.slurm

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source slurm/_common.sh

echo
echo "########## 1/2  Inference: load the trained checkpoint, evaluate on the test set ##########"
python3 part3_cnn/dawnbench/predict.py \
    --data-root "$CIFAR_ROOT" \
    --ckpt part3_cnn/dawnbench/checkpoints/resnet18_cifar10.pth

echo
EPOCHS="${EPOCHS:-1}"
echo "########## 2/2  Training: ${EPOCHS} epoch(s) from scratch ##########"
python3 part3_cnn/dawnbench/train.py \
    --data-root "$CIFAR_ROOT" \
    --epochs "$EPOCHS" --batch-size 512 --no-save
# --no-save still matters even now that the run is a full one. Saving triggers on
# acc > best_acc with best_acc starting from 0, so a demo run that happened to land
# slightly below the stored checkpoint would still replace it — and the inference step
# above depends on that file. The demo should never be able to damage its own evidence.
