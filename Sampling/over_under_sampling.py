import numpy as np
import matplotlib.pyplot as plt


def tri(x: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.abs(x), 0.0, None)


def X_triangle(f: np.ndarray, B: float) -> np.ndarray:
    return tri(f / B)


def replicas(f: np.ndarray, fs: float, B: float, kmax: int):
    ks = np.arange(-kmax, kmax + 1)
    shifted = f[None, :] - (ks[:, None] * fs)
    comps = tri(shifted / B)
    return comps, comps.sum(axis=0), ks


def kmax_for_span(fmax: float, fs: float) -> int:
    return int(np.ceil(fmax / fs)) + 1


def plot_replicas(ax, f, fs, B, title, fmax, overlay_base=False):
    kmax = kmax_for_span(fmax, fs)
    comps, summ, _ = replicas(f, fs, B, kmax)

    for c in comps:
        ax.plot(f, c, linewidth=1.0, alpha=0.30)
    ax.plot(f, summ, linewidth=2.5)

    if overlay_base:
        ax.plot(f, X_triangle(f, B), linestyle="--", linewidth=2.0)

    ax.axvline(+fs / 2, linestyle="--", linewidth=1.2)
    ax.axvline(-fs / 2, linestyle="--", linewidth=1.2)

    ax.set_title(title)
    ax.set_ylabel("Amplitude (arb.)")
    ax.grid(True)


def main():
    B = 1000.0
    fs_base = 3000.0
    fs_over = 6000.0
    fs_down = 1500.0
    B_aa = 700.0

    fmax = 9000.0
    f = np.linspace(-fmax, fmax, 12001)

    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)

    axes[0].plot(f, X_triangle(f, B), linewidth=2.5)
    axes[0].set_title(f"1) Original X(f): x(t)=sinc²(Bt), B={B:g} Hz")
    axes[0].set_ylabel("Amplitude (arb.)")
    axes[0].grid(True)

    plot_replicas(
        axes[1], f, fs_base, B,
        f"2) Sampling fs={fs_base:g} Hz (no overlap, fs/2={fs_base/2:g} > B)",
        fmax
    )

    plot_replicas(
        axes[2], f, fs_over, B,
        f"3) Oversampling fs={fs_over:g} Hz (replicas farther apart)",
        fmax
    )

    plot_replicas(
        axes[3], f, fs_down, B,
        f"4) Downsampling fs={fs_down:g} Hz (aliasing, fs/2={fs_down/2:g} < B)",
        fmax,
        overlay_base=True
    )

    plot_replicas(
        axes[4], f, fs_down, B_aa,
        f"5) With anti-alias bandlimit B_aa={B_aa:g} Hz (safe for fs/2={fs_down/2:g} Hz)",
        fmax,
        overlay_base=True
    )

    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(-fmax, fmax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()