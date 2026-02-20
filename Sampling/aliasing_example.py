import numpy as np
import matplotlib.pyplot as plt

f_max = 10
fs = 14

f = np.linspace(-20, 20, 1000)

def get_spectrum(f_shift):
    return np.maximum(0, 1 - np.abs(f - f_shift) / f_max)

S0 = get_spectrum(0)
S_plus1 = get_spectrum(fs)
S_minus1 = get_spectrum(-fs)

S_total = S0 + S_plus1 + S_minus1

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

axes[0].plot(f, S0, color='blue', label='Original Baseband Spectrum')
axes[0].fill_between(f, 0, S0, color='blue', alpha=0.3)
axes[0].set_title('1. Original Signal Spectrum ($f_{max}$ = 10 Hz)')
axes[0].set_xlim(-20, 20)
axes[0].set_ylim(0, 1.5)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].set_ylabel('Magnitude')
axes[0].legend()

axes[1].plot(f, S0, color='blue', label='Baseband ($0$)')
axes[1].plot(f, S_plus1, color='red', linestyle='--', label=f'Replica ($+f_s$, {fs} Hz)')
axes[1].plot(f, S_minus1, color='green', linestyle='--', label=f'Replica ($-f_s$, -{fs} Hz)')
axes[1].fill_between(f, 0, np.minimum(S0, S_plus1), color='red', alpha=0.3)
axes[1].fill_between(f, 0, np.minimum(S0, S_minus1), color='green', alpha=0.3)
axes[1].set_title(f'2. Overlapping Spectra ($f_s$ = {fs} Hz < 2*$f_{max}$)')
axes[1].set_xlim(-20, 20)
axes[1].set_ylim(0, 1.5)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].set_ylabel('Magnitude')
axes[1].legend()

axes[2].plot(f, S_total, color='purple', label='Summed (Observed) Spectrum')
axes[2].fill_between(f, 0, S_total, color='purple', alpha=0.3)
axes[2].set_title('3. Corrupted Spectrum (High-frequency components added to Baseband)')
axes[2].set_xlim(-20, 20)
axes[2].set_ylim(0, 1.5)
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Magnitude')
axes[2].legend()

plt.tight_layout()
plt.show()