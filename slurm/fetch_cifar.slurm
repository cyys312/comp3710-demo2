#!/bin/bash
#SBATCH --job-name=fetch-cifar
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/fetch-cifar-%j.out
#SBATCH --error=logs/fetch-cifar-%j.err

# Only downloads and unpacks CIFAR-10 into data/; takes no GPU.
# Why this is split out into a job of its own:
#   * The login node must not be used for downloads (it hurts everyone else, and the
#     admins will name you for it)
#   * The cluster gets only ~50 KB/s to the outside world, so 170 MB takes nearly an
#     hour; folded into a GPU training job that would leave the GPU idle on the network.
# Submit: sbatch slurm/fetch_cifar.slurm

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
echo "CIFAR-10 ready:"
du -sh data/cifar-10-batches-py
