#!/bin/bash
#SBATCH --job-name=demo-dawnbench
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/demo-dawnbench-%j.out
#SBATCH --error=logs/demo-dawnbench-%j.err

# Part 3.2 演示专用 —— 任务书要求：
#   "The model must run inference and a single epoch of training on the
#    Ranpur compute cluster during the demonstration." (1 Mark)
#
# 这个脚本一次做完两件事：用已训练的 checkpoint 跑推理，再跑一个 epoch 的训练。
# 提交： sbatch slurm/demo_dawnbench.slurm

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source slurm/_common.sh

echo
echo "########## 1/2  推理：载入已训练的 checkpoint，在测试集上评估 ##########"
python3 part3_cnn/dawnbench/predict.py \
    --data-root "$CIFAR_ROOT" \
    --ckpt part3_cnn/dawnbench/checkpoints/resnet18_cifar10.pth

echo
echo "########## 2/2  训练：从头跑一个 epoch，展示训练管线可用 ##########"
python3 part3_cnn/dawnbench/train.py \
    --data-root "$CIFAR_ROOT" \
    --epochs 1 --batch-size 512 --no-save
# --no-save 很重要：不加的话这一个 epoch 的模型会覆盖掉上面推理用的
# 那个 94.15% 的 checkpoint（保存条件是 acc > best_acc，而 best_acc 从 0 开始）
