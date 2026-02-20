import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

scale = 0.8

B = 10
f_stop = 12
fs_tight = 22

fs_sim = 1000
f = np.linspace(0, 40, 4000)

ideal_filter = (f <= B).astype(float)

rp_real = 1
rs_real = 30.0

N_real, Wn_real = signal.ellipord(B, f_stop, rp_real, rs_real, fs=fs_sim)
sos_real = signal.ellip(N_real, rp_real, rs_real, Wn_real, btype="low", fs=fs_sim, output="sos")

w_real, h_real = signal.sosfreqz(sos_real, worN=f, fs=fs_sim)
h_real_mag = np.abs(h_real)

pb_min = 10 ** (-rp_real / 20.0)
sb_max = 10 ** (-rs_real / 20.0)

idx_pb = np.where(h_real_mag >= pb_min)[0]
fp_eff = w_real[idx_pb[-1]] if idx_pb.size else np.nan

if np.isfinite(fp_eff):
    idx_after = np.where(w_real >= fp_eff)[0]
    idx_sb = idx_after[np.where(h_real_mag[idx_after] <= sb_max)[0]]
    fs_eff = w_real[idx_sb[0]] if idx_sb.size else np.nan
else:
    fs_eff = np.nan

omega = 2 * np.pi * f / fs_sim
phase = np.unwrap(np.angle(h_real))
with np.errstate(divide="ignore", invalid="ignore"):
    gd_real = -np.gradient(phase, omega)
finite = np.isfinite(gd_real)
if np.any(finite):
    gd_real[~finite] = gd_real[finite][0]
gd_real = np.maximum(gd_real, 0)

S0 = np.maximum(0, 1 - f / B)
S_rep = np.maximum(0, 1 - np.abs(f - fs_tight) / B)

duration = 0.4
t = np.arange(0, duration, 1 / fs_sim)
t0 = 0.08
x = (t >= t0).astype(float)
y = signal.sosfilt(sos_real, x)

fig, axes = plt.subplots(5, 1, figsize=(10 * scale, 18 * scale), dpi=90)

axes[0].plot(f, ideal_filter, linewidth=2)
axes[0].set_title("1. Ideal Filter")
axes[0].set_xlim(0, 40)
axes[0].set_ylim(0, 1.2)
axes[0].grid(True, linestyle="--", alpha=0.6)

axes[1].plot(f, ideal_filter, linewidth=2, label="Ideal Response")
axes[1].plot(w_real, h_real_mag, linewidth=2, label=f"Real Elliptic IIR (N={N_real})")

if np.isfinite(fp_eff) and np.isfinite(fs_eff) and fs_eff >= fp_eff:
    axes[1].axvspan(fp_eff, fs_eff, alpha=0.25)
    axes[1].text(
        (fp_eff + fs_eff) / 2,
        0.5,
        f"Effective\nWidth = {fs_eff - fp_eff:.2f} Hz",
        fontsize=11,
        ha="center",
        fontweight="bold",
    )
    axes[1].axvline(fp_eff, color="k", linestyle=":", linewidth=1, alpha=0.8)
    axes[1].axvline(fs_eff, color="k", linestyle=":", linewidth=1, alpha=0.8)
else:
    axes[1].axvspan(B, f_stop, alpha=0.15)
    axes[1].text(
        (B + f_stop) / 2,
        0.5,
        f"Spec\nWidth = {f_stop - B:.2f} Hz",
        fontsize=11,
        ha="center",
        fontweight="bold",
    )
    axes[1].axvline(B, color="k", linestyle=":", linewidth=1, alpha=0.8)
    axes[1].axvline(f_stop, color="k", linestyle=":", linewidth=1, alpha=0.8)

axes[1].set_title("2. Real Filter & Transition Band")
axes[1].set_xlim(0, 40)
axes[1].set_ylim(0, 1.2)
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].legend()

axes[2].plot(w_real, h_real_mag, linewidth=2, label="Real Filter Response")
axes[2].plot(f, S0, alpha=0.35, label="Baseband (mock)")
axes[2].plot(f, S_rep, linestyle="--", linewidth=2, label=f"High-freq Replica (mock, $f_s={fs_tight}$)")
axes[2].fill_between(f, 0, np.minimum(S_rep, h_real_mag), alpha=0.6, label="Aliasing Leakage (mock)")
axes[2].set_title("3. High-frequency Noise Leakage (Illustration)")
axes[2].set_xlim(0, 40)
axes[2].set_ylim(0, 1.2)
axes[2].grid(True, linestyle="--", alpha=0.6)
axes[2].legend()

axes[3].plot(f, gd_real, linewidth=2)
axes[3].axvline(B, color="k", linestyle=":")
axes[3].set_title(f"4. Group Delay of Real Filter (N={N_real})")
axes[3].set_xlim(0, 40)
axes[3].set_ylabel("Delay (samples)")
axes[3].grid(True, linestyle="--", alpha=0.6)

axes[4].plot(t, x, linestyle="--", alpha=0.35, label="Original (Step)")
axes[4].plot(t, y, linewidth=1.5, label="Filtered (Ringing)")
axes[4].set_title("5. Time Domain: Ringing Effect")
axes[4].set_xlim(0, duration)
axes[4].grid(True, linestyle="--", alpha=0.6)
axes[4].legend()

plt.tight_layout()
plt.show()