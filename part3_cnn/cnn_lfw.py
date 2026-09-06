"""
Part 3.1 — CNN classifier for LFW faces (PyTorch)

Requirement: two 3x3 convolution layers with 32 filters each, feeding a fully connected
             classifier; it should beat the Part 2 Eigenfaces + random forest baseline.

Run: python part3_cnn/cnn_lfw.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
SEED = 42


class SimpleCNN(nn.Module):
    """Two 3x3 / 32-filter convolution layers plus a fully connected classification head."""

    def __init__(self, n_classes: int, in_shape: tuple):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)

        # A dummy forward pass recovers the flattened dimension, so it never has to be hand-derived
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *in_shape)
            flat_dim = self._features(dummy).flatten(1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def _features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._features(x).flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def load_data():
    """Load LFW, normalise to [0,1] and reshape into 4D tensors (N, C, H, W)."""
    lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw.images.shape
    X = lfw.images.astype(np.float32)
    X = (X - X.min()) / (X.max() - X.min())      # normalise
    X = X[:, None, :, :]                          # add channel dim -> 4D
    y = lfw.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return train_ds, test_ds, lfw.target_names, (h, w)


def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(DEVICE))
            preds.append(out.argmax(1).cpu())
            trues.append(yb)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


def main():
    torch.manual_seed(SEED)
    train_ds, test_ds, target_names, in_shape = load_data()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = SimpleCNN(len(target_names), in_shape).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    print(f"Device: {DEVICE} | classes: {len(target_names)} | input: {in_shape}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            running += loss.item() * xb.size(0)

        y_pred, y_true = evaluate(model, test_loader)
        acc = (y_pred == y_true).mean()
        print(f"epoch {epoch:3d} | loss {running / len(train_ds):.4f} | test acc {acc:.4f}")

    y_pred, y_true = evaluate(model, test_loader)
    print("\n--- CNN on LFW ---")
    print(classification_report(y_true, y_pred, target_names=target_names))
    # TODO: compare F1 against the Part 2 random forest, and explain in the demo why the CNN wins


if __name__ == "__main__":
    main()
