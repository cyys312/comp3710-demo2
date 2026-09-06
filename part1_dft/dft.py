"""
Part 1 of 4 — Discrete Fourier Transform

Task sheet requirements:
  1. Reconstruct a square wave from a Fourier series of odd harmonics and observe the effect
     of adding more harmonics (Gibbs phenomenon).
  2. Rewrite square_wave / square_wave_fourier / naive_dft with PyTorch tensor operations.
  3. Additionally provide a naive DFT that runs explicitly on the GPU (no built-in FFT),
     compare its runtime with the NumPy naive DFT and the NumPy FFT, and explain why the
     fastest one is the fastest.
  4. Vary the problem size N and observe how the ranking of the three changes.

Run:
    python part1_dft/dft.py                  # full pipeline (with plots)
    python part1_dft/dft.py --sizes 256 1024 4096
"""

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")           # produces plots on headless machines (cluster / CI) too
import matplotlib.pyplot as plt  # noqa: E402  must be imported after use("Agg")

HERE = Path(__file__).parent
OUTDIR = HERE / "outputs"

# ------------------------------ Defaults ------------------------------
N_DEFAULT = 2048     # number of samples
T = 1.0              # signal duration (seconds)
F0 = 1.0             # square wave fundamental frequency (Hz)
HARMONICS = [1, 3, 5, 20, 50]   # harmonic counts used for the reconstruction comparison


def pick_device() -> torch.device:
    """Prefer CUDA (Rangpur A100), then Apple MPS, and fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """GPU kernels launch asynchronously; without this sync you time the launch, not the work."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# =====================================================================
# 1. Signal generation — NumPy and PyTorch versions
# =====================================================================
def square_wave(t: np.ndarray) -> np.ndarray:
    """Ideal square wave (NumPy)."""
    return np.sign(np.sin(2.0 * np.pi * F0 * t))


def square_wave_torch(t: torch.Tensor, f0: float = F0) -> torch.Tensor:
    """Ideal square wave (PyTorch). Pointwise identical to the NumPy version, tensor ops only."""
    return torch.sign(torch.sin(2.0 * torch.pi * f0 * t))


def square_wave_fourier(t: np.ndarray, f0: float, n_harmonics: int) -> np.ndarray:
    """Fourier series approximation of a square wave (NumPy): odd harmonics, 1/n amplitudes."""
    result = np.zeros_like(t)
    for k in range(n_harmonics):
        n = 2 * k + 1                       # odd harmonics 1, 3, 5, ...
        result += np.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result


def square_wave_fourier_torch(t: torch.Tensor, f0: float, n_harmonics: int) -> torch.Tensor:
    """Fourier series approximation of a square wave (PyTorch).

    The Python for loop is replaced by broadcasting: the harmonic orders n are laid out as a
    column, outer-multiplied with the time axis t and summed over the harmonic dimension, so
    every harmonic is computed in one go.
    """
    n = torch.arange(1, 2 * n_harmonics, 2, device=t.device, dtype=t.dtype)   # 1,3,5,...
    # (n_harmonics, 1) * (1, N) -> (n_harmonics, N), then sum along the harmonic dimension
    terms = torch.sin(2 * torch.pi * n[:, None] * f0 * t[None, :]) / n[:, None]
    return (4 / torch.pi) * terms.sum(dim=0)


# =====================================================================
# 2. DFT implementations
# =====================================================================
def naive_dft_loops(x: np.ndarray) -> np.ndarray:
    """Textbook naive DFT: two nested Python loops, strictly O(N^2).

    Only run for very small N; it exists to show that two implementations of the same O(N^2)
    algorithm can still differ by a factor of several hundred.
    """
    n_samples = len(x)
    out = np.zeros(n_samples, dtype=np.complex128)
    for k in range(n_samples):
        for n in range(n_samples):
            out[k] += x[n] * np.exp(-2j * np.pi * k * n / n_samples)
    return out


def _dft_matrix_np(n_samples: int) -> np.ndarray:
    """DFT matrix W[k,n] = e^{-2j*pi*k*n/N} (NumPy)."""
    idx = np.arange(n_samples)
    return np.exp(-2j * np.pi * np.outer(idx, idx) / n_samples)


def naive_dft(x: np.ndarray) -> np.ndarray:
    """Naive DFT (NumPy, matrix form).

    Mathematically just the matrix-vector product X = W x, still O(N^2) complex multiply-adds,
    but the loops now sit inside BLAS, which is two to three orders of magnitude faster than
    the nested Python loops.
    """
    return _dft_matrix_np(len(x)) @ x


def naive_dft_torch(x: torch.Tensor) -> torch.Tensor:
    """Naive DFT (PyTorch, DFT matrix built explicitly, no call to torch.fft).

    The whole computation happens on whatever device x lives on — pass a GPU tensor and this
    is the GPU naive DFT. It is still O(N^2), but the N^2 multiply-adds are spread over
    thousands of GPU cores and run in parallel.
    """
    n_samples = x.shape[-1]
    idx = torch.arange(n_samples, device=x.device, dtype=torch.float32)
    angle = -2.0 * torch.pi * idx[:, None] * idx[None, :] / n_samples
    # Build the complex matrix from cos/sin to avoid depending on a backend complex exp
    w = torch.complex(torch.cos(angle), torch.sin(angle))
    return w @ x.to(torch.complex64)


# =====================================================================
# 3. Timing benchmark
# =====================================================================
def _time_it(fn, repeats: int, device: torch.device | None = None) -> float:
    """Run repeats times and keep the minimum (more robust to system noise than the mean)."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if device is not None:
            sync(device)
        best = min(best, time.perf_counter() - start)
    return best


def benchmark(sizes, device: torch.device, repeats: int = 3):
    """Time the four methods for every N; returns {method name: [time, ...]}."""
    methods = ["NumPy naive DFT", f"Torch naive DFT ({device.type})",
               "NumPy FFT", f"Torch FFT ({device.type})"]
    results = {m: [] for m in methods}

    for n_samples in sizes:
        t_np = np.linspace(0.0, T, n_samples, endpoint=False)
        signal_np = square_wave_fourier(t_np, F0, 50)
        signal_gpu = torch.from_numpy(signal_np).float().to(device)

        # Warm-up: the first call includes kernel compilation and memory allocation, so it
        # must not be timed
        naive_dft_torch(signal_gpu); sync(device)
        torch.fft.fft(signal_gpu); sync(device)

        results[methods[0]].append(_time_it(lambda: naive_dft(signal_np), repeats))
        results[methods[1]].append(_time_it(lambda: naive_dft_torch(signal_gpu), repeats, device))
        results[methods[2]].append(_time_it(lambda: np.fft.fft(signal_np), repeats))
        results[methods[3]].append(_time_it(lambda: torch.fft.fft(signal_gpu), repeats, device))

    return results


def print_benchmark(sizes, results) -> None:
    """Print the timing table and a fastest-to-slowest ranking for each N."""
    names = list(results)
    width = max(len(n) for n in names) + 2
    print("\n--- DFT timing comparison (seconds, best of 3) ---")
    print(" " * width + "".join(f"N={n:<12d}" for n in sizes))
    for name in names:
        row = "".join(f"{v:<14.6f}" for v in results[name])
        print(f"{name:<{width}}{row}")

    print("\n--- Ranking from fastest to slowest for each N ---")
    for i, n_samples in enumerate(sizes):
        order = sorted(names, key=lambda m: results[m][i])
        print(f"N={n_samples:<6d} " + "  <  ".join(f"{m} ({results[m][i]*1e3:.2f} ms)"
                                                   for m in order))


# =====================================================================
# 4. Plotting
# =====================================================================
def plot_reconstructions(t, square, path):
    """Harmonics 1 to 50: the fit keeps improving, but the Gibbs overshoot at the jumps stays."""
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
    print("Saved:", path)


def plot_spectrum(t, signal, dft_result, n_samples, path):
    """Time-domain signal and its single-sided magnitude spectrum: peaks at odd harmonics only."""
    xf = np.fft.fftfreq(n_samples, d=T / n_samples)[: n_samples // 2]
    magnitude = 2.0 / n_samples * np.abs(dft_result[: n_samples // 2])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(t, signal, color="tab:cyan")
    ax1.set_title("Time domain: square wave (50 harmonics)")
    ax1.set_xlabel("Time [s]")
    ax1.grid(True)

    ax2.stem(xf[:60], magnitude[:60], basefmt=" ")
    for k in range(1, 12, 2):        # mark where the first few odd harmonics sit
        ax2.axvline(k * F0, color="r", ls="--", alpha=0.4)
    ax2.set_title("Frequency domain: naive DFT magnitude (odd harmonics only)")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("Saved:", path)


def plot_timings(sizes, results, path):
    """Runtimes on log-log axes: the slope gap between O(N^2) and O(N log N) is obvious."""
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
    print("Saved:", path)


# =====================================================================
# 5. Main pipeline
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Part 1: DFT / FFT")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096],
                        help="signal lengths used for the benchmark")
    parser.add_argument("--loop-n", type=int, default=512,
                        help="size of the double-loop naive DFT run (very slow, keep it small)")
    args = parser.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    device = pick_device()
    print(f"PyTorch {torch.__version__} | device = {device}")

    # ---- (a) square wave reconstruction ----
    t = np.linspace(0.0, T, N_DEFAULT, endpoint=False)
    square = square_wave(t)
    plot_reconstructions(t, square, OUTDIR / "square_wave_reconstruction.png")

    # ---- (b) agreement between the NumPy and PyTorch implementations ----
    t_torch = torch.from_numpy(t).float().to(device)
    max_diff_wave = (square_wave_torch(t_torch).cpu().numpy() - square).max()
    sig_np = square_wave_fourier(t, F0, 50)
    sig_torch = square_wave_fourier_torch(t_torch, F0, 50).cpu().numpy()
    print("\n--- NumPy vs PyTorch implementation agreement ---")
    print(f"square_wave         max deviation: {abs(max_diff_wave):.3e}")
    print(f"square_wave_fourier max deviation: {np.abs(sig_torch - sig_np).max():.3e}")

    dft_np = naive_dft(sig_np)
    dft_gpu = naive_dft_torch(torch.from_numpy(sig_np).float().to(device)).cpu().numpy()
    fft_np = np.fft.fft(sig_np)
    # Judge the error in relative terms: the spectral peaks are around 4N/pi ~ 1e3, so an
    # absolute error of 0.2 is really only a ~1e-4 relative error.
    scale = np.abs(fft_np).max()
    rel_np = np.abs(dft_np - fft_np).max() / scale
    rel_gpu = np.abs(dft_gpu - fft_np).max() / scale
    print(f"naive_dft(NumPy, complex128) vs np.fft.fft: relative error {rel_np:.2e}  "
          f"allclose={np.allclose(dft_np, fft_np)}")
    print(f"naive_dft(GPU,  complex64)  vs np.fft.fft: relative error {rel_gpu:.2e}  "
          f"(unavoidable when accumulating N terms in single precision, still within the "
          f"~1e-7*sqrt(N) float32 range)")

    plot_spectrum(t, sig_np, dft_np, N_DEFAULT, OUTDIR / "dft_spectrum.png")

    # ---- (c) how slow the double-loop version really is ----
    small = sig_np[: args.loop_n]
    start = time.perf_counter()
    loop_result = naive_dft_loops(small)
    loop_time = time.perf_counter() - start
    matrix_time = _time_it(lambda: naive_dft(small), 3)
    print(f"\n--- Same O(N^2), different implementations (N={args.loop_n}) ---")
    print(f"Nested Python loops : {loop_time:.4f} s")
    print(f"NumPy matrix form   : {matrix_time:.6f} s  ({loop_time / matrix_time:.0f}x faster)")
    print(f"Results agree       : {np.allclose(loop_result, naive_dft(small))}")

    # ---- (d) size sweep ----
    results = benchmark(args.sizes, device)
    print_benchmark(args.sizes, results)
    plot_timings(args.sizes, results, OUTDIR / "dft_timing.png")


if __name__ == "__main__":
    main()
