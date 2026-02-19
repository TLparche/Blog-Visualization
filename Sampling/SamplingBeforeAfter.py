import numpy as np
import matplotlib.pyplot as plt


def plot_sampling_process():
    # 1. 신호 파라미터 설정
    f = 1.0  # 신호의 주파수 (1 Hz)
    duration = 2.0  # 신호 길이 (2초)

    # 2. 연속 시간 신호 생성 (컴퓨터 시뮬레이션을 위해 매우 촘촘한 간격 사용)
    # 실제 연속 신호처럼 보이게 하기 위해 샘플링 주파수를 매우 높게 설정 (1000 Hz)
    fs_continuous = 1000
    t_continuous = np.linspace(0, duration, int(fs_continuous * duration))
    x_continuous = np.sin(2 * np.pi * f * t_continuous)

    # 3. 이산 시간 신호 생성 (샘플링)
    # 샘플링 주기 Ts = 0.1초 (즉, 샘플링 주파수 fs = 10 Hz)
    fs_discrete = 10
    Ts = 1 / fs_discrete

    # np.arange를 사용하여 0초부터 duration까지 Ts 간격으로 시간 배열 생성
    t_discrete = np.arange(0, duration, Ts)
    x_discrete = np.sin(2 * np.pi * f * t_discrete)

    # 4. 시각화
    plt.figure(figsize=(12, 6))

    # 연속 신호 그리기 (파란 실선)
    plt.plot(t_continuous, x_continuous, label='Continuous Signal $x(t)$', color='blue', alpha=0.6, linewidth=2)

    # 샘플링된 이산 신호 그리기 (빨간 점과 수직선)
    # stem 함수는 이산 신호를 표현하는 표준적인 방법입니다.
    plt.stem(t_discrete, x_discrete, linefmt='r--', markerfmt='ro', basefmt='k', label='Discrete Samples $x[n]$')

    # 그래프 설정
    plt.title('Sampling: Continuous to Discrete Signal Conversion')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.axhline(0, color='black', linewidth=1)  # x축 강조

    # 파일 저장 및 출력
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_sampling_process()