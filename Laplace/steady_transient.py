import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

wn = 25.0
zeta = 0.25
sys = signal.TransferFunction([wn**2], [1, 2 * zeta * wn, wn**2])

t = np.linspace(0, 2.0, 12000)
tout, y = signal.step(sys, T=t)

y_final = y[-1]
band = 0.02 * abs(y_final) if y_final != 0 else 0.02

within = np.abs(y - y_final) <= band
tail_ok = np.logical_and.accumulate(within[::-1])[::-1]

t_settle = None
if tail_ok.any():
    idx = np.argmax(tail_ok)
    t_settle = tout[idx]

plt.figure(figsize=(9, 4.5))
plt.plot(tout, y, linewidth=2)

if t_settle is not None:
    plt.axvspan(0.0, t_settle, color="orange", alpha=0.35, label="과도상태")
    plt.axvspan(t_settle, tout[-1], color="green", alpha=0.25, label="정상상태")
    plt.axhline(y_final + band, linestyle="--", linewidth=1)
    plt.axhline(y_final - band, linestyle="--", linewidth=1)
else:
    plt.axvspan(0.0, tout[-1], color="orange", alpha=0.25, label="과도상태")

plt.title("정상상태 vs 과도상태")
plt.xlabel("시간 (s)")
plt.ylabel("출력 y(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()