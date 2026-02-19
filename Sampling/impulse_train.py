import numpy as np
import matplotlib.pyplot as plt

def run_sampling_demo():
    f = 1.0
    duration = 2.0
    fs_high = 1000
    fs = 10
    Ts = 1 / fs

    t = np.linspace(0, duration, int(fs_high * duration))
    x = np.sin(2 * np.pi * f * t)

    t_sample = np.arange(0, duration, Ts)
    impulse = np.ones_like(t_sample)
    x_sample = np.sin(2 * np.pi * f * t_sample)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)

    ax1.plot(t, x, 'b', lw=1.5, label='$x(t)$')
    ax1.set_title(r'Continuous Signal $x(t)$')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlabel('Time (s)')
    ax1.legend(loc='upper right')

    markerline, stemlines, baseline = ax2.stem(
        t_sample, impulse,
        linefmt='k-', markerfmt='k^', basefmt='k-'
    )
    plt.setp(markerline, markersize=8)
    ax2.set_title(r'Impulse Train $p(t) = \sum \delta(t - nT)$')
    ax2.set_ylabel('Impulse')
    ax2.set_ylim(0, 1.5)

    tick_labels = ['0'] + ['T'] + [f'{k}T' for k in range(2, len(t_sample))]
    ax2.set_xticks(t_sample)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel('Time (nT)')

    ax3.plot(t, x, 'b--', alpha=0.3, label='Envelope')
    ax3.stem(t_sample, x_sample, linefmt='r-', markerfmt='ro', basefmt='k-')
    ax3.set_title(r'Sampled Output $x_s(t)$')
    ax3.set_ylabel('Amplitude')
    ax3.set_xlabel('Time (s)')
    ax3.legend(loc='upper right')

    for ax in [ax1, ax2, ax3]:
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sampling_demo()