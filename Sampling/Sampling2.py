# Sampling2.py
# Story-cut viewer: 1 scene = 1 main plot + caption
# Added: original signal scene + pre-FFT detection (envelope/threshold) scene
#
# Controls:
#   - Left/Right arrow: change scene
#   - S key: save current scene png
#   - Q/Esc: quit
#
# CLI:
#   python Sampling2.py
#   python Sampling2.py --scene 0
#   python Sampling2.py --save_all
#
# Requirements: numpy, matplotlib

import argparse
import textwrap
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# ---------------------------
# Font / Korean rendering
# ---------------------------
def setup_korean_font():
    candidates = [
        "Malgun Gothic",          # Windows
        "AppleGothic",            # macOS
        "NanumGothic",            # Linux common
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------
# Signal generation (analog-ish)
# ---------------------------
def gaussian_burst(t: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def make_original_signal(
    fs_view: float,
    T: float,
    f_main: float = 43_000.0,
    f_ghost: float = 170_000.0,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    'Original' continuous-time-like signal x_cont(t) sampled at fs_view (very high).
    Ground truth: main (43k) + high-frequency ghost (170k) that causes aliasing when sampled poorly.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, 1.0 / fs_view)

    burst = gaussian_burst(t, center=0.065, width=0.015)  # 65ms center
    env = 0.65 + 0.35 * (1.0 + 0.7 * burst)

    # Ghost grows during burst (front-end resonance / unknown-tech noise)
    ghost_amp = 0.10 + 0.55 * burst

    slow = 0.05 * np.sin(2 * np.pi * 3.0 * t)

    phase1 = 0.2
    phase2 = -0.9

    x = env * (
        1.00 * np.sin(2 * np.pi * f_main * t + phase1)
        + ghost_amp * np.sin(2 * np.pi * f_ghost * t + phase2)
    )
    x += slow
    x += 0.02 * rng.standard_normal(len(t))

    meta = {
        "f_main": f_main,
        "f_ghost": f_ghost,
        "burst_center_s": 0.065,
        "burst_width_s": 0.015,
    }
    return t, x, meta


# ---------------------------
# FIR lowpass (for AA / envelope)
# ---------------------------
def lowpass_fir(x: np.ndarray, fs: float, cutoff_hz: float, numtaps: int = 801) -> np.ndarray:
    if cutoff_hz <= 0:
        return x.copy()
    if numtaps % 2 == 0:
        numtaps += 1

    fc = cutoff_hz / fs  # normalized
    n = np.arange(numtaps) - (numtaps - 1) / 2
    h = 2 * fc * np.sinc(2 * fc * n)
    h *= np.hamming(numtaps)
    h /= np.sum(h)
    return np.convolve(x, h, mode="same")


# ---------------------------
# Sampling + FFT
# ---------------------------
def alias_to_baseband(f_hz: float, Fs_hz: float) -> float:
    f_mod = f_hz % Fs_hz
    if f_mod > Fs_hz / 2:
        return Fs_hz - f_mod
    return f_mod


def sample_signal(
    t_cont: np.ndarray,
    x_cont: np.ndarray,
    Fs: float,
    jitter_std_s: float = 0.0,
    seed: int = 0,
    T: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if T is None:
        T = t_cont[-1]
    N = int(T * Fs)
    t_s = np.arange(N) / Fs

    if jitter_std_s > 0:
        t_s = t_s + rng.normal(0.0, jitter_std_s, size=t_s.shape)
        t_s = np.clip(t_s, t_cont[0], t_cont[-1])

    x_s = np.interp(t_s, t_cont, x_cont)
    return t_s, x_s


def spectrum_db(x: np.ndarray, Fs: float, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    N = len(x)

    if window.lower() in ("hann", "hanning"):
        w = np.hanning(N)
    else:
        w = np.ones(N)

    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(N, d=1.0 / Fs)

    mag = np.abs(X) + 1e-12
    mag_db = 20.0 * np.log10(mag)
    mag_db -= np.max(mag_db)
    return f, mag_db


# ---------------------------
# Pre-FFT detection (envelope + threshold)
# ---------------------------
def envelope_trace(x_cont: np.ndarray, fs: float, cutoff_hz: float = 600.0) -> np.ndarray:
    # "How strong is it?" trace: abs + very low LPF
    return lowpass_fir(np.abs(x_cont), fs=fs, cutoff_hz=cutoff_hz, numtaps=1601)


def robust_threshold(y: np.ndarray, k: float = 5.0) -> float:
    """
    Median + k * MAD (robust). Good for "detect suspicious bump" visualization.
    """
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-12
    return med + k * mad


def find_segments(mask: np.ndarray, t: np.ndarray, min_len_s: float = 0.006) -> List[Tuple[float, float]]:
    """
    Return segments [t_start, t_end] where mask is True.
    """
    segs = []
    in_seg = False
    start_i = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            in_seg = True
            start_i = i
        if in_seg and (not v or i == len(mask) - 1):
            end_i = i if not v else i
            t0, t1 = t[start_i], t[end_i]
            if (t1 - t0) >= min_len_s:
                segs.append((t0, t1))
            in_seg = False
    return segs


# ---------------------------
# Scenes
# ---------------------------
@dataclass
class SceneCfg:
    key: str
    title: str
    caption: str
    mode: str  # "time", "detect", "spec"

    # Spec-mode config
    Fs: Optional[float] = None
    T_obs: Optional[float] = None
    aa_cutoff_hz: Optional[float] = None
    window: str = "hann"
    jitter_std_s: float = 0.0

    # View config
    xlim_ms: Tuple[float, float] = (0.0, 120.0)         # for time/detect
    xlim_khz: float = 120.0                             # for spec
    ylim_db: Tuple[float, float] = (-80.0, 5.0)         # for spec


def build_scenes(meta: Dict[str, float]) -> List[SceneCfg]:
    f_main = meta["f_main"]
    f_ghost = meta["f_ghost"]

    scenes = [
        SceneCfg(
            key="0",
            title="SCENE 0 — 원래 신호(연속시간 ‘진짜 세계’)",
            mode="time",
            caption=(
                "관측소에서 ‘실제로 존재하는’ 원 신호 파형(시간영역).\n"
                "여기서는 아직 FFT도, 샘플링도 안 함.\n"
                "→ 스토리상 리오가 ‘또… 이 패턴’이라고 느끼는 출발점."
            ),
            xlim_ms=(0.0, 120.0),
        ),
        SceneCfg(
            key="0B",
            title="SCENE 0B — FFT 전 탐지: 엔벨로프(강도 추세)로 ‘수상 구간’ 잡기",
            mode="detect",
            caption=(
                "FFT 전에 먼저 ‘언제 세졌는지’를 잡는 절차(=관측 절차).\n"
                "엔벨로프(진폭/에너지 추세)를 그리고 임계값을 넘는 구간을 표시.\n"
                "→ ‘어느 구간을 FFT/분석에 쓸지’가 이 단계에서 결정됨."
            ),
            xlim_ms=(0.0, 120.0),
        ),
        SceneCfg(
            key="1A",
            title="SCENE 1A — 리오의 ‘현장 FFT’ (샘플링 실수로 문제 발생)",
            mode="spec",
            caption=(
                "Fs=200kHz로 ‘일단 디지털화’한 뒤 FFT.\n"
                f"원래는 {int(f_main/1000)}kHz(메인)만 또렷해야 하는데,\n"
                f"{int(f_ghost/1000)}kHz(고역 성분)가 접혀서 ‘가짜 피크(alias)’가 생김."
            ),
            Fs=200_000.0,
            T_obs=0.12,
            aa_cutoff_hz=None,
            window="hann",
            jitter_std_s=0.0,
            xlim_khz=120.0,
        ),
        SceneCfg(
            key="1B",
            title="SCENE 1B — 히마리의 검증: Fs를 바꾸면 ‘가짜’가 움직인다",
            mode="spec",
            caption=(
                "Fs를 210kHz로 살짝 변경해 재측정.\n"
                "진짜 신호라면 피크 위치는 고정되어야 하지만,\n"
                "alias라면 샘플링 주파수에 따라 위치가 이동 → aliasing 확정."
            ),
            Fs=210_000.0,
            T_obs=0.12,
            aa_cutoff_hz=None,
            window="hann",
            jitter_std_s=0.0,
            xlim_khz=120.0,
        ),
        SceneCfg(
            key="2",
            title="SCENE 2 — 숙제①: Anti-aliasing(저역통과) 적용",
            mode="spec",
            caption=(
                "샘플링 전에 저역통과(AA 필터)로 고역 성분을 잘라냄.\n"
                "접힐 대상(고역)이 사라져서 ‘가짜 피크’가 약화/소실.\n"
                "→ ‘나이퀴스트만 지키면 끝’이 아니라, 전처리가 관측의 일부."
            ),
            Fs=200_000.0,
            T_obs=0.12,
            aa_cutoff_hz=80_000.0,
            window="hann",
            jitter_std_s=0.0,
            xlim_khz=120.0,
        ),
        SceneCfg(
            key="5",
            title="SCENE 5 — 토키의 ‘의외의 실수’ (절차 누락: 짧은 관측 + Rect + 지터)",
            mode="spec",
            caption=(
                "관측 길이가 짧으면(20ms) 스펙트럼이 퍼지고(leakage),\n"
                "Rect 창은 더 심하게 만들고, 지터는 바닥을 올림.\n"
                "→ ‘신속’만 있고 ‘검증 가능’이 없으면 결과가 흐려진다."
            ),
            Fs=120_000.0,
            T_obs=0.02,
            aa_cutoff_hz=None,
            window="rect",
            jitter_std_s=120e-9,
            xlim_khz=60.0,
        ),
        SceneCfg(
            key="6",
            title="SCENE 6 — 리오의 프로토콜 v1: ‘신속·정확·검증 가능’",
            mode="spec",
            caption=(
                "Fs 여유 확보 + AA 필터 ON + 충분한 관측 길이 + (필요시) Fs 변화 검증.\n"
                "결과: 메인 성분이 선명하고, alias가 억제됨.\n"
                "→ 토키가 반복 수행할 수 있게 ‘절차’로 고정."
            ),
            Fs=240_000.0,
            T_obs=0.12,
            aa_cutoff_hz=80_000.0,
            window="hann",
            jitter_std_s=0.0,
            xlim_khz=120.0,
        ),
    ]
    return scenes


# ---------------------------
# Plotting
# ---------------------------
def render_caption(ax, caption: str):
    wrapped = "\n".join(textwrap.fill(line, width=62) for line in caption.split("\n"))
    ax.text(
        0.01,
        -0.28,
        wrapped,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.12, pad=0.55),
    )


def render_time(ax, t_cont, x_cont, meta, scene: SceneCfg):
    ax.clear()
    t_ms = t_cont * 1000.0
    ax.plot(t_ms, x_cont, linewidth=1.1)

    ax.set_title(scene.title)
    ax.set_xlabel("시간 (ms)")
    ax.set_ylabel("신호 값")
    ax.grid(True, alpha=0.25)

    ax.set_xlim(scene.xlim_ms[0], scene.xlim_ms[1])

    # Mark burst center region lightly (story cue)
    c = meta["burst_center_s"] * 1000.0
    w = meta["burst_width_s"] * 1000.0
    ax.axvspan(c - 2.0 * w, c + 2.0 * w, alpha=0.12)

    ax.text(0.99, 0.95, "※ 음영: 반응 구간(의심)", transform=ax.transAxes, ha="right", va="top", alpha=0.8)
    render_caption(ax, scene.caption)


def render_detect(ax, t_cont, x_cont, meta, scene: SceneCfg):
    ax.clear()
    t_ms = t_cont * 1000.0

    env = envelope_trace(x_cont, fs=FS_VIEW, cutoff_hz=600.0)
    thr = robust_threshold(env, k=5.0)

    mask = env > thr
    segs = find_segments(mask, t_cont, min_len_s=0.006)

    ax.plot(t_ms, env, linewidth=1.7, label="엔벨로프(강도 추세)")
    ax.axhline(thr, linestyle="--", linewidth=1.2, alpha=0.85, label="임계값(탐지 기준)")

    for (t0, t1) in segs:
        ax.axvspan(t0 * 1000.0, t1 * 1000.0, alpha=0.18)

    ax.set_title(scene.title)
    ax.set_xlabel("시간 (ms)")
    ax.set_ylabel("강도(상대)")
    ax.grid(True, alpha=0.25)

    ax.set_xlim(scene.xlim_ms[0], scene.xlim_ms[1])
    ax.legend(loc="upper right", framealpha=0.2)

    ax.text(
        0.01,
        0.92,
        f"탐지 구간 수: {len(segs)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        alpha=0.85,
    )

    render_caption(ax, scene.caption)


def render_spec(ax, t_cont, x_cont, meta, scene: SceneCfg, seed_base: int = 123):
    assert scene.Fs is not None and scene.T_obs is not None

    f_main = meta["f_main"]
    f_ghost = meta["f_ghost"]

    # Anti-aliasing filter if requested
    x_src = x_cont
    if scene.aa_cutoff_hz is not None:
        x_src = lowpass_fir(x_src, fs=FS_VIEW, cutoff_hz=scene.aa_cutoff_hz, numtaps=801)

    # Sample
    _, x_s = sample_signal(
        t_cont=t_cont,
        x_cont=x_src,
        Fs=scene.Fs,
        jitter_std_s=scene.jitter_std_s,
        seed=seed_base + (hash(scene.key) % 10_000),
        T=scene.T_obs,
    )

    # Spectrum
    freqs, mag_db = spectrum_db(x_s, Fs=scene.Fs, window=scene.window)

    ax.clear()
    ax.plot(freqs / 1000.0, mag_db, linewidth=1.6)

    ax.set_title(scene.title)
    ax.set_xlabel("주파수 (kHz)")
    ax.set_ylabel("크기 (dB, peak=0)")
    ax.grid(True, alpha=0.25)

    ax.set_xlim(0.0, scene.xlim_khz)
    ax.set_ylim(scene.ylim_db[0], scene.ylim_db[1])

    nyq = scene.Fs / 2.0
    ax.axvline(nyq / 1000.0, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(
        nyq / 1000.0,
        scene.ylim_db[0] + 6,
        f"Nyquist={nyq/1000:.1f}k",
        rotation=90,
        va="bottom",
        ha="right",
        alpha=0.8,
    )

    ax.axvline(f_main / 1000.0, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        f_main / 1000.0,
        scene.ylim_db[0] + 6,
        f"main {f_main/1000:.1f}k",
        rotation=90,
        va="bottom",
        ha="right",
        alpha=0.85,
    )

    f_alias = alias_to_baseband(f_ghost, scene.Fs)
    if f_alias <= nyq + 1e-6:
        ax.axvline(f_alias / 1000.0, linestyle=":", linewidth=2.0, alpha=0.9)
        # ax.text(
        #     f_alias / 1000.0,
        #     scene.ylim_db[0] + 6,
        #     f"alias(ghost) {f_alias/1000:.1f}k",
        #     rotation=90,
        #     va="bottom",
        #     ha="right",
        #     alpha=0.9,
        # )

    render_caption(ax, scene.caption)


def render_scene(ax, t_cont, x_cont, meta, scene: SceneCfg):
    if scene.mode == "time":
        render_time(ax, t_cont, x_cont, meta, scene)
    elif scene.mode == "detect":
        render_detect(ax, t_cont, x_cont, meta, scene)
    elif scene.mode == "spec":
        render_spec(ax, t_cont, x_cont, meta, scene)
    else:
        raise ValueError(f"Unknown scene.mode: {scene.mode}")


def save_current(fig, scene_key: str):
    filename = f"scene_{scene_key}.png"
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    print(f"[saved] {filename}")


# ---------------------------
# Main
# ---------------------------
FS_VIEW = 2_000_000.0  # high-rate reference (analog-ish)


def main():
    setup_korean_font()

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="0", help="Start scene: 0, 0B, 1A, 1B, 2, 5, 6")
    parser.add_argument("--save_all", action="store_true", help="Save all scenes as PNGs then exit")
    args = parser.parse_args()

    T_total = 0.12
    t_cont, x_cont, meta = make_original_signal(fs_view=FS_VIEW, T=T_total)

    scenes = build_scenes(meta)
    scene_map: Dict[str, int] = {s.key: i for i, s in enumerate(scenes)}
    idx = scene_map.get(args.scene, 0)

    if args.save_all:
        fig, ax = plt.subplots(figsize=(11.5, 6.0))
        for s in scenes:
            render_scene(ax, t_cont, x_cont, meta, s)
            fig.tight_layout()
            fig.savefig(f"scene_{s.key}.png", dpi=180, bbox_inches="tight")
            print(f"[saved] scene_{s.key}.png")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(11.5, 6.0))

    def redraw():
        nonlocal idx
        render_scene(ax, t_cont, x_cont, meta, scenes[idx])
        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal idx
        if event.key in ("right", "d", "pagedown"):
            idx = (idx + 1) % len(scenes)
            redraw()
        elif event.key in ("left", "a", "pageup"):
            idx = (idx - 1) % len(scenes)
            redraw()
        elif event.key in ("s",):
            save_current(fig, scenes[idx].key)
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    print("Controls: ←/→ to change scene | S to save | Q/Esc to quit")
    plt.show()


if __name__ == "__main__":
    main()
