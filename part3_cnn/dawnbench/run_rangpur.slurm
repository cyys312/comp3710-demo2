#!/bin/bash
#SBATCH --job-name=dawnbench
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/dawnbench-%j.out
#SBATCH --error=logs/dawnbench-%j.err

# Part 3.2 DAWNBench —— ResNet-18 / CIFAR-10
# 提交： sbatch part3_cnn/dawnbench/run_rangpur.slurm
# 目标： >94% 准确率，训练时间与 V100 上的 360 秒相当或更快。

# 集群实测（2026-09-02）——三处和常见写法不同，照抄网上的模板会挂：
#   1. 分区用 a100（AllowAccounts=ALL）。课程专用的 comp3710 分区目前
#      所有人的作业都卡在 PENDING(PartitionConfig)，账号关联还没开通。
#   2. **不要写 --mem**。所有 a100 节点的 RealMemory=1，
#      任何 --mem=<正数> 都永远满足不了，作业会无限期 pending。
#   3. CIFAR-10 集群上已有共享副本：/home/groups/cifar/CIFAR-10
#      （torchvision 的 cifar-10-batches-py 格式，直接指过去即可）。
#      千万别现下：torchvision 默认源 cs.toronto.edu 到集群只有 ~3 KB/s，
#      170 MB 要下十几个小时；而集群到 GitHub 有 2.5 MB/s，是源的问题不是网的问题。
#   4. 不需要 module load cuda：conda 环境 torch 里的 PyTorch 2.13+cu130
#      自带 CUDA 运行时，加载 module 的 cuda/12.2 反而可能冲突。

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
source slurm/_common.sh

python3 part3_cnn/dawnbench/train.py \
    --epochs 30 \
    --batch-size 512 \
    --num-workers 4 \
    --data-root "$CIFAR_ROOT"
