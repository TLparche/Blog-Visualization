import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["axes.unicode_minus"] = False
for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = f
        break

t = np.linspace(0, 2.0, 4000)

A1, f1 = 0.35, 3.0
A2, f2 = 0.20, 9.0

carrier = 1.0 + A1*np.cos(2*np.pi*f1*t) + A2*np.cos(2*np.pi*f2*t)
x = np.exp(2*t) * carrier

sigma1 = x * np.exp(-1*t)
sigma3 = x * np.exp(-3*t)

c0 = "#1f77b4"
c1 = "#ff7f0e"
c3 = "#2ca02c"

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(t, x, linewidth=2, color=c0)
axes[0].set_title("원래 신호")
axes[0].set_ylabel("진폭")
axes[0].grid(True)

axes[1].plot(t, sigma1, linewidth=2, color=c1)
axes[1].set_title("감쇠 적용: σ=1  →  $x(t)e^{-t}$")
axes[1].set_ylabel("진폭")
axes[1].grid(True)

axes[2].plot(t, sigma3, linewidth=2, color=c3)
axes[2].set_title("감쇠 적용: σ=3  →  $x(t)e^{-3t}$")
axes[2].set_xlabel("시간")
axes[2].set_ylabel("진폭")
axes[2].grid(True)

plt.tight_layout()
plt.show()