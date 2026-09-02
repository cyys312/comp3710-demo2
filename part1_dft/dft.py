"""
Part 1 of 4 —— 离散傅里叶变换 (Discrete Fourier Transform)

任务书要求：
  1. 用奇次谐波的傅里叶级数重建方波，观察谐波数增加的效果（Gibbs 现象）。
  2. 把 square_wave / square_wave_fourier / naive_dft 用 PyTorch 张量运算重写。
  3. 额外做一个显式跑在 GPU 上的 naive_dft（不许用内建 FFT），
     与 NumPy 朴素 DFT、NumPy FFT 比较耗时，并解释为什么最快的最快。
  4. 改变数据规模 N，观察三者耗时排序如何变化。

运行:
    python part1_dft/dft.py                  # 完整流程（含图）
    python part1_dft/dft.py --sizes 256 1024 4096
"""

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")           # 无显示环境（集群/CI）也能出图
import matplotlib.pyplot as plt  # noqa: E402  必须在 use("Agg") 之后导入

HERE = Path(__file__).parent
OUTDIR = HERE / "outputs"

# ------------------------------ 默认参数 ------------------------------
N_DEFAULT = 2048     # 采样点数
T = 1.0              # 信号时长（秒）
F0 = 1.0             # 方波基频 (Hz)
HARMONICS = [1, 3, 5, 20, 50]   # 重建对比用的谐波个数


def pick_device() -> torch.device:
    """优先 CUDA（Rangpur A100），其次 Apple MPS，最后退回 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """GPU 上的算子是异步下发的，计时前必须同步，否则测到的只是下发时间。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# =====================================================================
# 1. 信号生成 —— NumPy 版与 PyTorch 版
# =====================================================================
def square_wave(t: np.ndarray) -> np.ndarray:
    """理想方波（NumPy）。"""
    return np.sign(np.sin(2.0 * np.pi * F0 * t))


def square_wave_torch(t: torch.Tensor, f0: float = F0) -> torch.Tensor:
    """理想方波（PyTorch）。与 NumPy 版逐点等价，只是换成张量算子。"""
    return torch.sign(torch.sin(2.0 * torch.pi * f0 * t))


def square_wave_fourier(t: np.ndarray, f0: float, n_harmonics: int) -> np.ndarray:
    """方波的傅里叶级数近似（NumPy）：只含奇次谐波，幅度按 1/n 衰减。"""
    result = np.zeros_like(t)
    for k in range(n_harmonics):
        n = 2 * k + 1                       # 奇次谐波 1, 3, 5, ...
        result += np.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result


def square_wave_fourier_torch(t: torch.Tensor, f0: float, n_harmonics: int) -> torch.Tensor:
    """方波的傅里叶级数近似（PyTorch）。

    这里把 Python 的 for 循环换成了广播：把谐波次数 n 摆成一列，
    与时间 t 做外积后按谐波维求和，一次算完所有谐波。
    """
    n = torch.arange(1, 2 * n_harmonics, 2, device=t.device, dtype=t.dtype)   # 1,3,5,...
    # (n_harmonics, 1) * (1, N) -> (n_harmonics, N)，再沿谐波维求和
    terms = torch.sin(2 * torch.pi * n[:, None] * f0 * t[None, :]) / n[:, None]
    return (4 / torch.pi) * terms.sum(dim=0)


# =====================================================================
# 2. DFT 的几种实现
# =====================================================================
def naive_dft_loops(x: np.ndarray) -> np.ndarray:
    """教科书式的朴素 DFT：双重 Python 循环，严格 O(N^2)。

    只在很小的 N 上跑，用来说明「同样是 O(N^2)，实现方式差几百倍」。
    """
    n_samples = len(x)
    out = np.zeros(n_samples, dtype=np.complex128)
    for k in range(n_samples):
        for n in range(n_samples):
            out[k] += x[n] * np.exp(-2j * np.pi * k * n / n_samples)
    return out


def _dft_matrix_np(n_samples: int) -> np.ndarray:
    """DFT 矩阵 W[k,n] = e^{-2j*pi*k*n/N}（NumPy）。"""
    idx = np.arange(n_samples)
    return np.exp(-2j * np.pi * np.outer(idx, idx) / n_samples)


def naive_dft(x: np.ndarray) -> np.ndarray:
    """朴素 DFT（NumPy 矩阵形式）。

    数学上就是矩阵-向量乘 X = W x，仍是 O(N^2) 次复数乘加，
    但循环下沉到 BLAS，比双重 Python 循环快两三个数量级。
    """
    return _dft_matrix_np(len(x)) @ x


def naive_dft_torch(x: torch.Tensor) -> torch.Tensor:
    """朴素 DFT（PyTorch，显式构造 DFT 矩阵，不调用 torch.fft）。

    传入的 x 在哪个 device 上，整个计算就在哪个 device 上完成 ——
    传入 GPU 张量即为「GPU 版 naive DFT」。仍然是 O(N^2)，
    但 N^2 次乘加会被摊到数千个 GPU 核心上并行执行。
    """
    n_samples = x.shape[-1]
    idx = torch.arange(n_samples, device=x.device, dtype=torch.float32)
    angle = -2.0 * torch.pi * idx[:, None] * idx[None, :] / n_samples
    # 用 cos/sin 组装复数矩阵，避免依赖后端的复数指数算子
    w = torch.complex(torch.cos(angle), torch.sin(angle))
    return w @ x.to(torch.complex64)


# =====================================================================
# 3. 计时基准
# =====================================================================
def _time_it(fn, repeats: int, device: torch.device | None = None) -> float:
    """跑 repeats 次取最小值（最小值比均值更抗系统噪声）。"""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if device is not None:
            sync(device)
        best = min(best, time.perf_counter() - start)
    return best


def benchmark(sizes, device: torch.device, repeats: int = 3):
    """对每个 N 比较四种方法的耗时，返回 {方法名: [耗时,...]}。"""
    methods = ["NumPy naive DFT", f"Torch naive DFT ({device.type})",
               "NumPy FFT", f"Torch FFT ({device.type})"]
    results = {m: [] for m in methods}

    for n_samples in sizes:
        t_np = np.linspace(0.0, T, n_samples, endpoint=False)
        signal_np = square_wave_fourier(t_np, F0, 50)
        signal_gpu = torch.from_numpy(signal_np).float().to(device)

        # 预热：首次调用包含 kernel 编译/显存分配，不能计入
        naive_dft_torch(signal_gpu); sync(device)
        torch.fft.fft(signal_gpu); sync(device)

        results[methods[0]].append(_time_it(lambda: naive_dft(signal_np), repeats))
        results[methods[1]].append(_time_it(lambda: naive_dft_torch(signal_gpu), repeats, device))
        results[methods[2]].append(_time_it(lambda: np.fft.fft(signal_np), repeats))
        results[methods[3]].append(_time_it(lambda: torch.fft.fft(signal_gpu), repeats, device))

    return results


def print_benchmark(sizes, results) -> None:
    """打印耗时表，并对每个 N 给出「由快到慢」的排序。"""
    names = list(results)
    width = max(len(n) for n in names) + 2
    print("\n--- DFT 耗时对比（秒，取 3 次最小值）---")
    print(" " * width + "".join(f"N={n:<12d}" for n in sizes))
    for name in names:
        row = "".join(f"{v:<14.6f}" for v in results[name])
        print(f"{name:<{width}}{row}")

    print("\n--- 每个 N 下由快到慢的排序 ---")
    for i, n_samples in enumerate(sizes):
        order = sorted(names, key=lambda m: results[m][i])
        print(f"N={n_samples:<6d} " + "  <  ".join(f"{m} ({results[m][i]*1e3:.2f} ms)"
                                                   for m in order))


# =====================================================================
# 4. 绘图
# =====================================================================
def plot_reconstructions(t, square, path):
    """谐波数从 1 增到 50：逼近越来越好，但跳变处的过冲（Gibbs 现象）不会消失。"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(t, square, "k", label="Square wave")
    axes[0].set_title("Original Square Wave")
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    for ax, n_harm in zip(axes[1:], HARMONICS):
        ax.plot(t, square_wave_fourier(t, F0, n_harm), label=f"N={n_harm} harmonics")
        ax.plot(t, square, "k--", alpha=0.5, label="Square wave")
        ax.set_title(f"Fourier Approximation, N={n_harm}")
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("已保存:", path)


def plot_spectrum(t, signal, dft_result, n_samples, path):
    """时域信号与其单边幅度谱：应当只在奇次谐波处出现峰值。"""
    xf = np.fft.fftfreq(n_samples, d=T / n_samples)[: n_samples // 2]
    magnitude = 2.0 / n_samples * np.abs(dft_result[: n_samples // 2])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(t, signal, color="tab:cyan")
    ax1.set_title("Time domain: square wave (50 harmonics)")
    ax1.set_xlabel("Time [s]")
    ax1.grid(True)

    ax2.stem(xf[:60], magnitude[:60], basefmt=" ")
    for k in range(1, 12, 2):        # 标出前几个奇次谐波的位置
        ax2.axvline(k * F0, color="r", ls="--", alpha=0.4)
    ax2.set_title("Frequency domain: naive DFT magnitude (odd harmonics only)")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("已保存:", path)


def plot_timings(sizes, results, path):
    """双对数坐标下的耗时曲线：O(N^2) 与 O(N log N) 的斜率差一眼可见。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, times in results.items():
        ax.plot(sizes, times, "o-", label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (number of samples)")
    ax.set_ylabel("time [s]")
    ax.set_title("DFT implementations: runtime vs problem size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("已保存:", path)


# =====================================================================
# 5. 主流程
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Part 1: DFT / FFT")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096],
                        help="基准测试用的信号长度")
    parser.add_argument("--loop-n", type=int, default=512,
                        help="双重循环版 naive DFT 的规模（很慢，别设大）")
    args = parser.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    device = pick_device()
    print(f"PyTorch {torch.__version__} | device = {device}")

    # ---- (a) 方波重建 ----
    t = np.linspace(0.0, T, N_DEFAULT, endpoint=False)
    square = square_wave(t)
    plot_reconstructions(t, square, OUTDIR / "square_wave_reconstruction.png")

    # ---- (b) NumPy 与 PyTorch 实现的一致性检查 ----
    t_torch = torch.from_numpy(t).float().to(device)
    max_diff_wave = (square_wave_torch(t_torch).cpu().numpy() - square).max()
    sig_np = square_wave_fourier(t, F0, 50)
    sig_torch = square_wave_fourier_torch(t_torch, F0, 50).cpu().numpy()
    print("\n--- NumPy vs PyTorch 实现一致性 ---")
    print(f"square_wave         最大偏差: {abs(max_diff_wave):.3e}")
    print(f"square_wave_fourier 最大偏差: {np.abs(sig_torch - sig_np).max():.3e}")

    dft_np = naive_dft(sig_np)
    dft_gpu = naive_dft_torch(torch.from_numpy(sig_np).float().to(device)).cpu().numpy()
    fft_np = np.fft.fft(sig_np)
    # 误差要看相对量：谱峰幅度约 4N/pi ~ 1e3，绝对误差 0.2 其实只有 ~1e-4 的相对误差。
    scale = np.abs(fft_np).max()
    rel_np = np.abs(dft_np - fft_np).max() / scale
    rel_gpu = np.abs(dft_gpu - fft_np).max() / scale
    print(f"naive_dft(NumPy, complex128) vs np.fft.fft: 相对误差 {rel_np:.2e}  "
          f"allclose={np.allclose(dft_np, fft_np)}")
    print(f"naive_dft(GPU,  complex64)  vs np.fft.fft: 相对误差 {rel_gpu:.2e}  "
          f"(单精度累加 N 项的必然结果，仍在 float32 的 ~1e-7*sqrt(N) 量级内)")

    plot_spectrum(t, sig_np, dft_np, N_DEFAULT, OUTDIR / "dft_spectrum.png")

    # ---- (c) 双重循环版有多慢 ----
    small = sig_np[: args.loop_n]
    start = time.perf_counter()
    loop_result = naive_dft_loops(small)
    loop_time = time.perf_counter() - start
    matrix_time = _time_it(lambda: naive_dft(small), 3)
    print(f"\n--- 同为 O(N^2)，实现方式的差距 (N={args.loop_n}) ---")
    print(f"双重 Python 循环 : {loop_time:.4f} s")
    print(f"NumPy 矩阵形式   : {matrix_time:.6f} s  ({loop_time / matrix_time:.0f}x 更快)")
    print(f"两者结果一致     : {np.allclose(loop_result, naive_dft(small))}")

    # ---- (d) 规模扫描 ----
    results = benchmark(args.sizes, device)
    print_benchmark(args.sizes, results)
    plot_timings(args.sizes, results, OUTDIR / "dft_timing.png")


if __name__ == "__main__":
    main()
