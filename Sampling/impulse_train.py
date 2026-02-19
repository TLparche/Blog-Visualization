import numpy as np
import matplotlib.pyplot as plt


def plot_impulse_train():
    # 설정
    T = 1
    n_range = 5
    n = np.arange(-n_range, n_range + 1)
    t = n * T

    plt.figure(figsize=(10, 5))

    # Impulse Train 그리기 (화살표 스타일)
    markerline, stemlines, baseline = plt.stem(
        t, np.ones_like(t),
        linefmt='k-', markerfmt='k^', basefmt='k-'
    )

    # 스타일 조정
    plt.setp(markerline, markersize=10, markerfacecolor='black')
    plt.setp(stemlines, linewidth=1.5)
    plt.setp(baseline, visible=False)  # x축 바닥선 숨기기 (취향)

    # 축 설정
    plt.axhline(0, color='black', linewidth=1)
    plt.ylim(0, 1.5)
    plt.yticks([0, 1])

    # X축 라벨링 (-2T, -T, 0, T, 2T ...)
    xtick_labels = [f'{i}T' if i != 0 else '0' for i in n]
    plt.xticks(t, xtick_labels, fontsize=11)

    plt.title(r'Impulse Train $p(t) = \sum \delta(t - kT)$', fontsize=14)
    plt.xlabel('Time (t)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_impulse_train()