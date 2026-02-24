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

base = np.exp(2*t) * carrier

sigmas = [1.0, 2.0, 2.2, 3.0]
colors = ["#ff7f0e", "#d62728", "#9467bd", "#2ca02c"]

fig, ax = plt.subplots(figsize=(10, 4))

for s, c in zip(sigmas, colors):
    y = base * np.exp(-s*t)
    ax.plot(t, y, linewidth=2, color=c, label=f"σ={s}")

ax.set_ylim(0, 6)

ax.set_title("ROC 직관: σ > 2에서 수렴")
ax.set_xlabel("시간 t")
ax.set_ylabel("진폭")
ax.legend()
ax.grid(True)

ax.text(
    0.02, 0.97,
    r"$f(t)e^{-\sigma t}$"
    "\n"
    r"$\sigma=2$는 경계, $\sigma>2$에서 감쇠",
    transform=ax.transAxes,
    va="top"
)

plt.tight_layout()
plt.show()