import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

tau = 1.0
T_init = 2.0


def get_time_signal(t, T, tau):
    t_mod = (t + T / 2) % T - T / 2
    return np.where(np.abs(t_mod) <= tau / 2, 1, 0)


def get_freq_spectrum(T, tau, n_harmonics=30):
    f0 = 1 / T
    ns = np.arange(-n_harmonics, n_harmonics + 1)
    freqs = ns * f0

    c_n = (tau / T) * np.sinc(freqs * tau)
    return freqs, np.abs(c_n)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
plt.subplots_adjust(bottom=0.2, hspace=0.8)

t = np.linspace(-10, 10, 1000)
f_cont = np.linspace(-5, 5, 500)

x_t = get_time_signal(t, T_init, tau)
line1, = ax1.plot(t, x_t, lw=2, color='blue')
ax1.set_title(f'Time Domain (Period T = {T_init:.2f})')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.set_ylim(-0.2, 1.2)
ax1.grid(True, alpha=0.3)

freqs, mags = get_freq_spectrum(T_init, tau)

env2, = ax2.plot(f_cont, (tau/T_init)*np.abs(np.sinc(f_cont*tau)), 'r--', alpha=0.5, label='Envelope')
ax2.stem(freqs, mags, basefmt="k-", markerfmt="bo")

ax2.set_title('2. Fourier Coefficients (|Cn|)\n(Amplitude decreases as T increases)')
ax2.set_ylabel('|Cn|')
ax2.set_xlim(-5, 5)
ax2.set_ylim(0, 0.6)
ax2.grid(True, alpha=0.3)

env3, = ax3.plot(f_cont, tau * np.abs(np.sinc(f_cont*tau)), 'g--', alpha=0.5, lw=2, label='Envelope (Fixed)')
ax3.stem(freqs, mags * T_init, basefmt="k-", markerfmt="go")

ax3.set_title('3. Scaled Coefficients (T * |Cn|)\n(Converges to Fourier Transform X(f))')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('T * |Cn|')
ax3.set_xlim(-5, 5)
ax3.set_ylim(0, 1.2)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
s_period = Slider(ax_slider, 'Period (T)', 1, 30, valinit=T_init, valstep=0.5)

def update(val):
    current_T = s_period.val

    new_x = get_time_signal(t, current_T, tau)
    line1.set_ydata(new_x)
    ax1.set_title(f'Time Domain (Period T = {current_T:.2f})')

    new_freqs, new_mags = get_freq_spectrum(current_T, tau)

    ax2.cla()
    ax2.plot(f_cont, (tau / current_T) * np.abs(np.sinc(f_cont * tau)), 'r--', alpha=0.5)
    ax2.stem(new_freqs, new_mags, basefmt="k-", markerfmt="bo")

    ax2.set_title('2. Fourier Coefficients (|Cn|) - Amplitude Shrinks')
    ax2.set_ylabel('|Cn|')
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(0, 0.6)
    ax2.grid(True, alpha=0.3)

    ax3.cla()
    ax3.plot(f_cont, tau * np.abs(np.sinc(f_cont * tau)), 'g--', alpha=0.5, lw=2, label='Fourier Transform')
    ax3.stem(new_freqs, new_mags * current_T, basefmt="k-", markerfmt="go")

    ax3.set_title('3. Scaled Coefficients (T * |Cn|) - Shape Preserved')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('T * |Cn|')
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(0, 1.2)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

    fig.canvas.draw_idle()

s_period.on_changed(update)

plt.show()