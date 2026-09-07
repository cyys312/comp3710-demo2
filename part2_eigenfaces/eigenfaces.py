"""
Part 2 of 4 —— Eigenfaces (face PCA) + random forest classification

Pipeline:
  LFW faces -> mean-centre -> SVD -> keep the first n_components eigenvectors (eigenfaces)
  -> project train/test onto the "face space" -> random forest -> classification report

Run: python part2_eigenfaces/eigenfaces.py
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")            # Works without a display; every figure goes to disk
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import fetch_lfw_people
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

N_COMPONENTS = 150
RANDOM_STATE = 42
OUTDIR = Path(__file__).parent / "outputs"


def load_lfw(min_faces_per_person: int = 70, resize: float = 0.4):
    """Download and load LFW (the first run pulls roughly 200 MB over the network)."""
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize)
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names

    print("Total dataset size:")
    print(f"  n_samples : {n_samples}")
    print(f"  n_features: {X.shape[1]}")
    print(f"  n_classes : {target_names.shape[0]}")
    return X, y, target_names, h, w


def compute_pca(X_train, X_test, n_components=N_COMPONENTS):
    """Mean-centre the training set, run SVD, return the components and the projections.

    The mean must come from the training set alone and then be subtracted from both splits;
    computing it over all the data leaks test information into the model.
    """
    mean = np.mean(X_train, axis=0)
    X_train = X_train - mean
    X_test = X_test - mean

    U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
    components = Vt[:n_components]

    X_train_pca = X_train @ components.T
    X_test_pca = X_test @ components.T
    return components, S, X_train_pca, X_test_pca, mean


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Lay out a set of flattened faces as a gallery; returns the figure so it can be saved."""
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(min(n_row * n_col, len(images))):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=9)
        plt.xticks(())
        plt.yticks(())
    return fig


def plot_compactness(S):
    """Compactness curve: variance explained by the first k components; is the basis compact?"""
    explained = (S ** 2) / np.sum(S ** 2)
    fig = plt.figure()
    plt.plot(np.cumsum(explained) * 100)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance [%]")
    plt.title("PCA compactness")
    plt.grid(True)
    cum = np.cumsum(explained)
    for k in (50, 100, 150):
        print(f"  The first {k:3d} components explain {cum[k - 1] * 100:.2f}% of the variance")
    return fig


def plot_reconstruction_progression(mean, components, target, h, w, path):
    """Rebuild one face from the mean plus an increasing number of eigenfaces.

    This is the clearest way to see what an eigenface is: the top row is the running
    reconstruction, the bottom row is the single eigenface added at that step. The first few
    components turn out to adjust overall illumination rather than facial structure, which is
    why the reconstruction barely changes until k is about 5.
    """
    ks = [0, 1, 2, 3, 5, 8, 15, 30, 60, len(components)]
    coefficients = (target - mean) @ components.T

    fig, axes = plt.subplots(2, len(ks) + 1, figsize=(1.3 * (len(ks) + 1), 3.6))
    for col, k in enumerate(ks):
        recon = mean + components[:k].T @ coefficients[:k] if k else mean.copy()
        axes[0, col].imshow(recon.reshape(h, w), cmap=plt.cm.gray)
        axes[0, col].set_title(f"k = {k}" if k else "mean face", fontsize=8)
        if k:
            axes[1, col].imshow(components[k - 1].reshape(h, w), cmap=plt.cm.gray)
            axes[1, col].set_title(f"eigenface #{k}", fontsize=7, color="0.45")
        else:
            axes[1, col].axis("off")
    axes[0, -1].imshow(target.reshape(h, w), cmap=plt.cm.gray)
    axes[0, -1].set_title("original", fontsize=8, color="C3")
    axes[1, -1].axis("off")
    for ax in axes.ravel():
        ax.set_xticks(())
        ax.set_yticks(())
    axes[0, 0].set_ylabel("reconstruction", fontsize=8)
    fig.suptitle("Top: mean face + weighted sum of the first k eigenfaces      "
                 "Bottom: the k-th eigenface on its own", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=155, bbox_inches="tight")
    print("saved:", path)


def main():
    X, y, target_names, h, w = load_lfw()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    components, S, X_train_pca, X_test_pca, mean = compute_pca(X_train, X_test)
    eigenfaces = components.reshape((N_COMPONENTS, h, w))
    print("Projected shapes:", X_train_pca.shape, X_test_pca.shape)

    OUTDIR.mkdir(exist_ok=True)
    plot_gallery(eigenfaces, [f"eigenface {i}" for i in range(len(eigenfaces))], h, w) \
        .savefig(OUTDIR / "eigenfaces.png", dpi=120)
    plot_compactness(S).savefig(OUTDIR / "compactness.png", dpi=120)
    # X_test still holds the raw (not mean-centred) test images, so row 3 is a real face
    plot_reconstruction_progression(mean, components, X_test[3], h, w,
                                    OUTDIR / "reconstruction_progression.png")

    clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)

    acc = (y_pred == y_test).mean()
    print("\n--- Random Forest on PCA features ---")
    print(f"Test accuracy          : {acc:.4f}  ({int((y_pred == y_test).sum())}/{len(y_test)})")

    # Majority-class baseline: George W Bush alone accounts for about 41% of LFW, so an
    # accuracy quoted without this baseline badly overstates what the model is worth.
    majority = np.bincount(y_test).max() / len(y_test)
    print(f"Majority-class baseline: {majority:.4f}  (always predict "
          f"{target_names[np.bincount(y_test).argmax()]})")
    print(f"Gain over baseline     : {acc - majority:+.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Prediction gallery: check against ground truth to see which people the errors land on
    titles = [
        f"pred: {target_names[p].split()[-1]}\ntrue: {target_names[t].split()[-1]}"
        for p, t in zip(y_pred, y_test)
    ]
    plot_gallery(X_test, titles, h, w).savefig(OUTDIR / "predictions.png", dpi=120)
    print(f"Figures saved to {OUTDIR}")

    ablation(X_train_pca, X_test_pca, y_train, y_test)


def ablation(X_train_pca, X_test_pca, y_train, y_test):
    """Two controlled experiments that explain what the 57% figure actually means.

    1. Number of retained components —— the compactness curve says the first 50 already
       explain 85% of the variance, so how much accuracy is lost cutting 150 dims to 50?
    2. class_weight="balanced" —— the confusion matrix above shows the model assigning
       almost every sample to the majority class. Does weighting the minority classes
       improve macro-F1?
       (Accuracy may drop instead: majority-class hits traded for minority-class recall.)
    """
    print("\n--- Ablation 1: effect of the number of components on accuracy ---")
    for k in (25, 50, 100, 150):
        clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train_pca[:, :k], y_train)
        acc_k = (clf.predict(X_test_pca[:, :k]) == y_test).mean()
        print(f"  first {k:3d} components: accuracy {acc_k:.4f}")

    print("\n--- Ablation 2: class weighting as compensation for class imbalance ---")
    for weight in (None, "balanced"):
        clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE,
                                     n_jobs=-1, class_weight=weight)
        clf.fit(X_train_pca, y_train)
        pred = clf.predict(X_test_pca)
        acc_w = (pred == y_test).mean()
        macro_f1 = f1_score(y_test, pred, average="macro")
        print(f"  class_weight={str(weight):8s}: accuracy {acc_w:.4f} | macro-F1 {macro_f1:.4f}")


if __name__ == "__main__":
    main()
