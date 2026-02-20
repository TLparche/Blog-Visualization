import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def run_sampling_demo():
    B = 4.0
    duration = 2.0
    fs_high = 1000
    f_max = 25

    t = np.linspace(0, duration, int(fs_high * duration))
    x = np.sinc(B * (t - duration / 2)) ** 2

    f_cont = np.linspace(-f_max, f_max, 2000)
    X_f = np.maximum(0, 1 - np.abs(f_cont) / B)

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    plt.subplots_adjust(bottom=0.15, hspace=0.5, wspace=0.25)

    axs[0, 0].plot(t, x, 'b', lw=1.5)
    axs[0, 0].set_title(r'Continuous Signal $x(t) = \mathrm{sinc}^2(B t)$')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylim(-0.2, 1.2)
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].axhline(0, color='black', linewidth=1)

    axs[0, 1].plot(f_cont, X_f, 'b-', lw=1.5)
    axs[0, 1].fill_between(f_cont, 0, X_f, color='blue', alpha=0.2)
    axs[0, 1].set_title(r'Spectrum $X(f)$')
    axs[0, 1].set_xlim(-f_max, f_max)
    axs[0, 1].set_ylim(0, 1.2)
    axs[0, 1].set_ylabel('Magnitude')
    axs[0, 1].set_xlabel('Frequency (Hz)')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].axhline(0, color='black', linewidth=1)

    def draw_dynamic(fs):
        for i in range(1, 3):
            for j in range(2):
                axs[i, j].clear()
                axs[i, j].grid(True, linestyle='--', alpha=0.6)
                axs[i, j].axhline(0, color='black', linewidth=1)

        Ts = 1 / fs
        t_sample = np.arange(0, duration, Ts)
        impulse = np.ones_like(t_sample)
        x_sample = np.sinc(B * (t_sample - duration / 2))**2

        k_max = int(f_max / fs) + 1
        k_indices = np.arange(-k_max, k_max + 1)
        freq_P = k_indices * fs
        amp_P = np.ones_like(freq_P) * fs

        X_s_f = np.zeros_like(f_cont)
        for k in k_indices:
            X_s_f += fs * np.maximum(0, 1 - np.abs(f_cont - k * fs) / B)

        markerline, stemlines, baseline = axs[1, 0].stem(
            t_sample, impulse, linefmt='k-', markerfmt='k^', basefmt='k-'
        )
        plt.setp(markerline, markersize=8)
        axs[1, 0].set_title(r'Impulse Train $p(t)$')
        axs[1, 0].set_ylabel('Impulse')
        axs[1, 0].set_ylim(0, 1.5)
        axs[1, 0].set_xlim(0, duration)
        axs[1, 0].set_xlabel('Time (s)')

        markerline, stemlines, baseline = axs[1, 1].stem(
            freq_P, amp_P, linefmt='k-', markerfmt='k^', basefmt='k-'
        )
        plt.setp(markerline, markersize=8)
        axs[1, 1].set_title(r'Impulse Train Spectrum $P(f)$')
        axs[1, 1].set_xlim(-f_max, f_max)
        axs[1, 1].set_ylim(0, 22)
        axs[1, 1].set_ylabel('Magnitude')
        axs[1, 1].set_xlabel('Frequency (Hz)')

        axs[2, 0].plot(t, x, 'b--', alpha=0.3)
        axs[2, 0].stem(t_sample, x_sample, linefmt='r-', markerfmt='ro', basefmt='k-')
        axs[2, 0].set_title(r'Sampled Output $x_s(t)$')
        axs[2, 0].set_ylabel('Amplitude')
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].set_ylim(-0.2, 1.2)
        axs[2, 0].set_xlim(0, duration)

        axs[2, 1].plot(f_cont, X_s_f, 'r-', lw=1.5)
        axs[2, 1].fill_between(f_cont, 0, X_s_f, color='red', alpha=0.2)
        axs[2, 1].set_title(r'Sampled Spectrum $X_s(f)$ ,$f_s$: '+ str(fs) + 'Hz' )
        axs[2, 1].set_xlim(-f_max, f_max)
        axs[2, 1].set_ylim(0, 22)
        axs[2, 1].set_ylabel('Magnitude')
        axs[2, 1].set_xlabel('Frequency (Hz)')

        fig.canvas.draw_idle()

    init_fs = 10.0
    draw_dynamic(init_fs)

    ax_slider = plt.axes((0.2, 0.05, 0.6, 0.03))
    fs_slider = Slider(
        ax=ax_slider,
        label='Sampling Rate ($f_s$)',
        valmin=3.0,
        valmax=20.0,
        valinit=init_fs,
        valstep=1.0
    )

    def update(val):
        draw_dynamic(fs_slider.val)

    fs_slider.on_changed(update)

    plt.show()

if __name__ == "__main__":
    run_sampling_demo()