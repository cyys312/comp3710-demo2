#!/bin/bash
#SBATCH --job-name=fetch-cifar
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/fetch-cifar-%j.out
#SBATCH --error=logs/fetch-cifar-%j.err

# 只负责把 CIFAR-10 下载并解包到 data/，不占 GPU。
# 单独拆成一个作业的原因：
#   * 登录节点不能跑下载（会影响所有人，也会被管理员点名）
#   * 集群到外网只有 ~50 KB/s，170 MB 要下近一小时；
#     混在 GPU 训练作业里会把 GPU 白白占着等网络。
# 提交： sbatch slurm/fetch_cifar.slurm

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs data
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

python3 - <<'PY'
import torchvision
for train in (True, False):
    ds = torchvision.datasets.CIFAR10("data", train=train, download=True)
    print(("train" if train else "test "), len(ds), flush=True)
PY
echo "CIFAR-10 就绪:"
du -sh data/cifar-10-batches-py
