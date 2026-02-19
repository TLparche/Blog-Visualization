import numpy as np
import matplotlib.pyplot as plt

def run_sampling_demo():
    B = 4.0
    duration = 2.0
    fs_high = 1000
    fs = 10
    Ts = 1 / fs
    f_max = 25

    t = np.linspace(0, duration, int(fs_high * duration))
    x = np.sinc(B * (t - duration / 2)) ** 2

    t_sample = np.arange(0, duration, Ts)
    impulse = np.ones_like(t_sample)
    x_sample = np.sinc(B * (t_sample - duration / 2)) ** 2

    f_cont = np.linspace(-f_max, f_max, 2000)
    X_f = np.maximum(0, 1 - np.abs(f_cont) / B)

    k_max = int(f_max / fs)
    k_indices = np.arange(-k_max, k_max + 1)
    freq_P = k_indices * fs
    amp_P = np.ones_like(freq_P) * fs

    X_s_f = np.zeros_like(f_cont)
    for k in k_indices:
        X_s_f += fs * np.maximum(0, 1 - np.abs(f_cont - k * fs) / B)

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.5, wspace=0.25)

    axs[0, 0].plot(t, x, 'b', lw=1.5)
    axs[0, 0].set_title(r'Continuous Signal $x(t) = \mathrm{sinc}^2(B t)$')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylim(-0.2, 1.2)

    axs[0, 1].plot(f_cont, X_f, 'b-', lw=1.5)
    axs[0, 1].fill_between(f_cont, 0, X_f, color='blue', alpha=0.2)
    axs[0, 1].set_title(r'Spectrum $X(f)$')
    axs[0, 1].set_xlim(-f_max, f_max)
    axs[0, 1].set_ylim(0, 1.2)
    axs[0, 1].set_ylabel('Magnitude')
    axs[0, 1].set_xlabel('Frequency (Hz)')

    markerline, stemlines, baseline = axs[1, 0].stem(
        t_sample, impulse, linefmt='k-', markerfmt='k^', basefmt='k-'
    )
    plt.setp(markerline, markersize=8)
    axs[1, 0].set_title(r'Impulse Train $p(t) = \sum \delta(t - nT)$')
    axs[1, 0].set_ylabel('Impulse')
    axs[1, 0].set_ylim(0, 1.5)

    tick_labels_t = ['0'] + ['T'] + [f'{k}T' for k in range(2, len(t_sample))]
    axs[1, 0].set_xticks(t_sample)
    axs[1, 0].set_xticklabels(tick_labels_t)
    axs[1, 0].set_xlabel('Time (nT)')

    markerline, stemlines, baseline = axs[1, 1].stem(
        freq_P, amp_P, linefmt='k-', markerfmt='k^', basefmt='k-'
    )
    plt.setp(markerline, markersize=8)
    axs[1, 1].set_title(r'Impulse Train Spectrum $P(f)$')
    axs[1, 1].set_xlim(-f_max, f_max)
    axs[1, 1].set_ylim(0, fs * 1.5)
    axs[1, 1].set_ylabel('Magnitude')

    tick_labels_f = [f'{k}f_s' if k != 0 else '0' for k in k_indices]
    axs[1, 1].set_xticks(freq_P)
    axs[1, 1].set_xticklabels(tick_labels_f)
    axs[1, 1].set_xlabel('Frequency ($k f_s$)')

    axs[2, 0].plot(t, x, 'b--', alpha=0.3)
    axs[2, 0].stem(t_sample, x_sample, linefmt='r-', markerfmt='ro', basefmt='k-')
    axs[2, 0].set_title(r'Sampled Output $x_s(t)$')
    axs[2, 0].set_ylabel('Amplitude')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylim(-0.2, 1.2)

    axs[2, 1].plot(f_cont, X_s_f, 'r-', lw=1.5)
    axs[2, 1].fill_between(f_cont, 0, X_s_f, color='red', alpha=0.2)
    axs[2, 1].set_title(r'Sampled Spectrum $X_s(f) = \frac{1}{T} \sum X(f - k f_s)$')
    axs[2, 1].set_xlim(-f_max, f_max)
    axs[2, 1].set_ylim(0, fs * 1.2)
    axs[2, 1].set_ylabel('Magnitude')
    axs[2, 1].set_xlabel('Frequency (Hz)')

    for ax in axs.flat:
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sampling_demo()