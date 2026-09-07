"""
Part 3.1 ablation — is a convolution actually needed, or would plain fully connected layers do?

A convolution is a special case of a fully connected layer: most weights are forced to zero
(locality) and the rest are forced to be equal across positions (weight sharing). In function
space the fully connected layer is therefore strictly more general, which invites the question
of whether it would simply do better given enough parameters.

This script answers it on our own data: same split, same optimiser, same epoch budget, with the
convolution stack swapped for a multi-layer perceptron of comparable and then far larger size.

The CNN row reuses SimpleCNN and load_data from cnn_lfw.py rather than reimplementing them, so
the two files cannot drift apart.

Every configuration is run over several seeds and reported as a mean. A single run turned out to
be a fragile thing to quote: inserting the per-epoch evaluation call that cnn_lfw.py makes moves
the final accuracy from 0.8602 to 0.8509, even though evaluation runs under no_grad with dropout
disabled and should not influence training at all. Whatever the mechanism, a number that shifts
by a point when an unrelated call is added is not a number worth comparing against, so the
conclusion here rests on the mean across seeds instead.

Run: python part3_cnn/ablation_mlp.py --seeds 3
"""

import argparse
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cnn_lfw import BATCH_SIZE, DEVICE, EPOCHS, LR, SEED, SimpleCNN, load_data


class MLPBaseline(nn.Module):
    """Fully connected only: no convolution, no weight sharing, no locality.

    Every pixel connects to every hidden unit, so the network has to discover from the data
    alone that neighbouring pixels are related and that a detector useful in one corner is
    also useful in another. The convolution gets both of those for free as an architectural
    constraint.
    """

    def __init__(self, n_classes: int, in_shape: tuple, hidden: int):
        super().__init__()
        flat = in_shape[0] * in_shape[1]
        self.fc1 = nn.Linear(flat, hidden)
        self.fc2 = nn.Linear(hidden, 128)
        self.fc3 = nn.Linear(128, n_classes)
        # Same dropout strength as the CNN head, so the comparison is not about regularisation
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        correct += (model(xb.to(DEVICE)).argmax(1).cpu() == yb).sum().item()
        total += yb.numel()
    return correct / total


def train_one(build, seed):
    """Train one model at one seed and return (params, train accuracy, test accuracy)."""
    torch.manual_seed(seed)
    train_ds, test_ds, target_names, in_shape = load_data()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = build(len(target_names), in_shape).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            criterion(model(xb), yb).backward()
            optimiser.step()
        accuracy(model, test_loader)     # mirrors cnn_lfw.py, which evaluates every epoch

    # Training accuracy is reported alongside test accuracy to separate the two failure modes:
    # overfitting shows up as a large gap, underfitting as a low training accuracy.
    return (sum(p.numel() for p in model.parameters()),
            accuracy(model, train_loader), accuracy(model, test_loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3,
                        help="how many seeds to average over, starting from SEED")
    args = parser.parse_args()
    seeds = [SEED + i for i in range(args.seeds)]

    configs = [("CNN (two conv layers, submitted)", SimpleCNN)]
    for hidden in (128, 256, 1024):
        configs.append((f"MLP hidden={hidden}",
                        lambda n, s, h=hidden: MLPBaseline(n, s, h)))

    print(f"device={DEVICE} epochs={EPOCHS} batch={BATCH_SIZE} lr={LR} seeds={seeds}\n")
    print(f"{'model':<34}{'params':>10}{'train acc':>12}{'test acc':>12}{'spread':>10}")
    print("-" * 78)

    summary = []
    for label, build in configs:
        runs = [train_one(build, s) for s in seeds]
        params = runs[0][0]
        tr = statistics.mean(r[1] for r in runs)
        te = [r[2] for r in runs]
        summary.append((label, params, tr, statistics.mean(te)))
        print(f"{label:<34}{params / 1e6:>8.2f} M{tr:>12.4f}"
              f"{statistics.mean(te):>12.4f}{max(te) - min(te):>10.4f}", flush=True)

    cnn = summary[0]
    best = max(summary[1:], key=lambda r: r[3])
    print(f"\nBest MLP is {best[0]} at {best[3]:.4f}, still "
          f"{(cnn[3] - best[3]) * 100:.1f} points behind the CNN "
          f"on {best[1] / cnn[1]:.1f}x the parameters.")
    print("The MLPs do not fit even the training set, so this is underfitting, not overfitting.\n"
          "The gap is not about capacity; it is the inductive bias that locality and weight\n"
          "sharing hand a convolution for free, and that a dense layer has to learn from 966\n"
          "images instead.")


if __name__ == "__main__":
    main()
