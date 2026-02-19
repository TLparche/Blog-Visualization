import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from matplotlib.gridspec import GridSpec

t_duration = 1.0
fs1_init = 8.0
fs2_init = 25.0
initial_text = r"\cos(2*\pi*5*t) + 0.5*\cos(2*\pi*15*t)"


def parse_and_generate(expression, t):
    expr = expression.replace(r'\sin', 'np.sin')
    expr = expr.replace(r'\cos', 'np.cos')
    expr = expr.replace(r'\pi', 'np.pi')
    expr = expr.replace(r'\exp', 'np.exp')
    expr = expr.replace('^', '**')
    expr = expr.replace(r'\cdot', '*')

    try:
        allowed = {"t": t, "np": np}
        result = eval(expr, {"__builtins__": {}}, allowed)
        if isinstance(result, (int, float)):
            result = np.full_like(t, result)
        return result, True
    except:
        return np.zeros_like(t), False


def compute_fft_precise(x, fs, n_fft=2048):
    if len(x) > 1:
        window = np.hanning(len(x))
        x_windowed = x * window
        win_sum = np.sum(window)
        if win_sum == 0: win_sum = 1
        scale_factor = 2.0 / win_sum
    else:
        x_windowed = x
        scale_factor = 2.0 / len(x) if len(x) > 0 else 1

    X = np.fft.fft(x_windowed, n=n_fft)
    X_real = np.real(X) * scale_factor
    X_mag = np.abs(X) * scale_factor


    freqs = np.fft.fftfreq(n_fft, d=1 / fs)
    shift_idx = np.argsort(freqs)

    return freqs[shift_idx], X_real[shift_idx], X_mag[shift_idx]


fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 3, figure=fig, height_ratios=[0.8, 1, 1], hspace=0.4, wspace=0.15)

ax_time = fig.add_subplot(gs[0, :])
ax_ov1 = fig.add_subplot(gs[1, 0])
ax_sum1 = fig.add_subplot(gs[1, 1])
ax_mag1 = fig.add_subplot(gs[1, 2])
ax_ov2 = fig.add_subplot(gs[2, 0])
ax_sum2 = fig.add_subplot(gs[2, 1])
ax_mag2 = fig.add_subplot(gs[2, 2])


def setup_ax(ax, title, mode="time"):
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(title, fontsize=11, pad=5)

    if mode == "time":
        ax.set_xlim(0, 0.4)
        ax.set_ylabel("Amplitude")
    elif mode == "real":
        ax.set_xlim(0, 45)
        ax.set_ylim(-1.2, 1.2)
        ax.set_ylabel("Real Part")
    elif mode == "mag":
        ax.set_xlim(0, 45)
        ax.set_ylim(0, 1.5)
        ax.set_ylabel("Magnitude (Abs)")

t_cont = np.arange(0, t_duration, 1 / 2000)


def update_plot(expression, fs1, fs2):
    y_cont, success = parse_and_generate(expression, t_cont)
    if not success: return

    t1 = np.arange(0, t_duration, 1 / fs1)
    y1, _ = parse_and_generate(expression, t1)

    t2 = np.arange(0, t_duration, 1 / fs2)
    y2, _ = parse_and_generate(expression, t2)

    ax_time.cla()
    setup_ax(ax_time, f"Time Domain Signal: ${expression}$", True)
    ax_time.plot(t_cont, y_cont, 'k-', alpha=0.4, label='Analog')
    ax_time.vlines(t1, 0, y1, colors='b', linestyles='-', alpha=0.5)
    ax_time.scatter(t1, y1, color='b', s=15, label=f'Fs1={fs1:.0f}Hz')
    ax_time.vlines(t2, 0, y2, colors='r', linestyles='-', alpha=0.5)
    ax_time.scatter(t2, y2, color='r', s=15, marker='x', label=f'Fs2={fs2:.0f}Hz')
    ax_time.legend(loc='upper right')

    f_high, r_high, m_high = compute_fft_precise(y_cont, 2000)
    f_real1, r_real1, m_real1 = compute_fft_precise(y1, fs1, 4096)
    f_real2, r_real2, m_real2 = compute_fft_precise(y2, fs2, 4096)

    def plot_freq_pair(ax_ov, ax_sum, ax_mag, fs, f_res, r_res, m_res, color, label_prefix):
        ax_ov.cla()
        ax_sum.cla()
        ax_mag.cla()

        setup_ax(ax_ov, f"[{label_prefix}] Overlap Process", "real")
        setup_ax(ax_sum, f"[{label_prefix}] Summed Result", "real")
        setup_ax(ax_mag, f"[{label_prefix}] Final Result", "mag")

        replicas = range(0, 4)
        for k in replicas:
            shifted_f = f_high + k * fs
            mask = (shifted_f >= -10) & (shifted_f <= 50)
            f_part = shifted_f[mask]
            r_part = r_high[mask]

            if len(f_part) == 0: continue

            if k == 0:
                ax_ov.plot(f_part, r_part, color='k', lw=1.5, alpha=0.8, label='Original')
                ax_ov.fill_between(f_part, 0, r_part, color='gray', alpha=0.2)
            else:
                ax_ov.plot(f_part, r_part, color=color, ls='--', lw=1, alpha=0.6)

        view_max = 50
        n_repeat = int(view_max / fs) + 2
        for k in range(n_repeat):
            shifted_f = f_res + k * fs
            sort_idx = np.argsort(shifted_f)
            f_plot = shifted_f[sort_idx]
            r_plot = r_res[sort_idx]
            mask = (f_plot >= 0) & (f_plot <= 50)
            ax_sum.plot(f_plot[mask], r_plot[mask], color=color, lw=1.5)

        mask_orig = (f_high >= 0) & (f_high <= 50)
        ax_sum.fill_between(f_high[mask_orig], 0, r_high[mask_orig], color='k', alpha=0.1, label='Original Position')

        for k in range(n_repeat):
            shifted_f = f_res + k * fs
            sort_idx = np.argsort(shifted_f)

            f_plot = shifted_f[sort_idx]
            m_plot = m_res[sort_idx]

            mask = (f_plot >= 0) & (f_plot <= 50)
            ax_mag.plot(f_plot[mask], m_plot[mask], color=color, lw=1.5)

        ax_mag.fill_between(f_high[mask_orig], 0, m_high[mask_orig], color='k', alpha=0.1, label='Original')
        ax_mag.legend(loc='upper right', fontsize=8)

    plot_freq_pair(ax_ov1, ax_sum1, ax_mag1, fs1, f_real1, r_real1, m_real1, 'b', 'Fs1')
    plot_freq_pair(ax_ov2, ax_sum2, ax_mag2, fs2, f_real2, r_real2, m_real2, 'r', 'Fs2')

    fig.canvas.draw_idle()


ax_box = plt.axes((0.15, 0.95, 0.5, 0.04))
ax_fs1 = plt.axes((0.75, 0.96, 0.2, 0.02))
ax_fs2 = plt.axes((0.75, 0.93, 0.2, 0.02))

text_box = TextBox(ax_box, 'Signal: ', initial=initial_text)
s_fs1 = Slider(ax_fs1, 'Fs1', 1.0, 50.0, valinit=fs1_init, valstep=1.0, valfmt='%0.0f Hz')
s_fs2 = Slider(ax_fs2, 'Fs2', 1.0, 50.0, valinit=fs2_init, valstep=1.0, valfmt='%0.0f Hz')


def submit(text): update_plot(text, s_fs1.val, s_fs2.val)


def update_s(val): update_plot(text_box.text, s_fs1.val, s_fs2.val)


text_box.on_submit(submit)
s_fs1.on_changed(update_s)
s_fs2.on_changed(update_s)

update_plot(initial_text, fs1_init, fs2_init)
plt.show()