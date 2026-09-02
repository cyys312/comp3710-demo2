"""
Part 2 of 4 —— Eigenfaces（人脸 PCA）+ 随机森林分类

流程：
  LFW 人脸 -> 去均值 -> SVD 特征分解 -> 取前 n_components 个特征向量（eigenfaces）
  -> 把训练/测试集投影到 "face space" -> 随机森林分类 -> 分类报告

运行: python part2_eigenfaces/eigenfaces.py
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")            # 无显示环境也能出图，图一律存盘
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import fetch_lfw_people
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

N_COMPONENTS = 150
RANDOM_STATE = 42
OUTDIR = Path(__file__).parent / "outputs"


def load_lfw(min_faces_per_person: int = 70, resize: float = 0.4):
    """下载并加载 LFW（首次运行会联网下载约 200 MB）。"""
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
    """对训练集去均值后做 SVD，返回主成分与投影结果。

    注意：均值必须只用训练集算，再同时减到测试集上，否则测试信息会污染模型。
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
    """把一组扁平化的人脸图排成画廊显示，返回 figure 以便存盘。"""
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
    """紧致度曲线：前 k 个成分解释了多少方差。用来判断降维是否够 compact。"""
    explained = (S ** 2) / np.sum(S ** 2)
    fig = plt.figure()
    plt.plot(np.cumsum(explained) * 100)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance [%]")
    plt.title("PCA compactness")
    plt.grid(True)
    cum = np.cumsum(explained)
    for k in (50, 100, 150):
        print(f"  前 {k:3d} 个成分解释了 {cum[k - 1] * 100:.2f}% 的方差")
    return fig


def main():
    X, y, target_names, h, w = load_lfw()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    components, S, X_train_pca, X_test_pca, _ = compute_pca(X_train, X_test)
    eigenfaces = components.reshape((N_COMPONENTS, h, w))
    print("Projected shapes:", X_train_pca.shape, X_test_pca.shape)

    OUTDIR.mkdir(exist_ok=True)
    plot_gallery(eigenfaces, [f"eigenface {i}" for i in range(len(eigenfaces))], h, w) \
        .savefig(OUTDIR / "eigenfaces.png", dpi=120)
    plot_compactness(S).savefig(OUTDIR / "compactness.png", dpi=120)

    clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)

    acc = (y_pred == y_test).mean()
    print("\n--- Random Forest on PCA features ---")
    print(f"测试集准确率: {acc:.4f}  ({int((y_pred == y_test).sum())}/{len(y_test)})")

    # 多数类基线：LFW 里 George W Bush 一个人就占了约 41%，
    # 不给出这个基线，单看准确率会高估模型的实际价值。
    majority = np.bincount(y_test).max() / len(y_test)
    print(f"多数类基线    : {majority:.4f}  (全部猜 "
          f"{target_names[np.bincount(y_test).argmax()]})")
    print(f"相对基线的提升: {acc - majority:+.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # 预测结果画廊：对照真值检查错在哪些人身上
    titles = [
        f"pred: {target_names[p].split()[-1]}\ntrue: {target_names[t].split()[-1]}"
        for p, t in zip(y_pred, y_test)
    ]
    plot_gallery(X_test, titles, h, w).savefig(OUTDIR / "predictions.png", dpi=120)
    print(f"图已保存到 {OUTDIR}")

    ablation(X_train_pca, X_test_pca, y_train, y_test)


def ablation(X_train_pca, X_test_pca, y_train, y_test):
    """两组对照实验，用来解释「57% 这个数字意味着什么」。

    1. 保留的主成分个数 —— 紧致度曲线说前 50 个成分已解释 85% 方差，
       那么把特征从 150 维砍到 50 维，分类准确率会掉多少？
    2. class_weight="balanced" —— 上面的混淆矩阵显示模型几乎把所有样本
       都判给了多数类。给少数类加权后 macro-F1 会不会变好？
       （注意准确率可能反而下降：牺牲多数类换少数类的召回。）
    """
    print("\n--- 对照实验 1：主成分个数对准确率的影响 ---")
    for k in (25, 50, 100, 150):
        clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(X_train_pca[:, :k], y_train)
        acc_k = (clf.predict(X_test_pca[:, :k]) == y_test).mean()
        print(f"  前 {k:3d} 个成分: 准确率 {acc_k:.4f}")

    print("\n--- 对照实验 2：类别加权对不平衡的补偿 ---")
    for weight in (None, "balanced"):
        clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE,
                                     n_jobs=-1, class_weight=weight)
        clf.fit(X_train_pca, y_train)
        pred = clf.predict(X_test_pca)
        acc_w = (pred == y_test).mean()
        macro_f1 = f1_score(y_test, pred, average="macro")
        print(f"  class_weight={str(weight):8s}: 准确率 {acc_w:.4f} | macro-F1 {macro_f1:.4f}")


if __name__ == "__main__":
    main()
