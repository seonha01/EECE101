"""
Week 2 둘러보기: 단계마다 내가 만든 함수가 어떻게 동작하는지 그림으로 확인한다.

    python explore_week2.py 1     # 단계 1: BFSK 파형, 스펙트럼 (+ 소리 파일)       (TODO 1 필요)
    python explore_week2.py 2     # 단계 2: 잡음 속에서 비트 읽기                    (TODO 2 필요)
    python explore_week2.py 3     # 단계 3: chirp 상관과 프레임 시작 찾기            (TODO 1, 3 필요)
    python explore_week2.py 4     # 단계 4: 헤더 다수결 (화면 출력)                 (TODO 4 필요)
    python explore_week2.py 5     # 단계 5: 프레임 구조 (화면 출력)                 (TODO 5 필요)

모뎀 옵션을 뒤에 붙여서 바꿔 볼 수 있다.  예) python explore_week2.py 3 --sync_thr 0.05
                                            python explore_week2.py 5 --frame_bits 64
그림은 results/explore_step*.png 로 저장된다.
"""
import os
import sys
import argparse
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modem

cfg = modem.add_modem_args(argparse.ArgumentParser()).parse_args(sys.argv[2:])
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
SPS = modem.samples_per_symbol(cfg)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print("그림 저장:", path)


def ref_tones(bits, c=cfg):
    # 비교/시험용 송신 신호 (비트마다 사인파를 따로 만들어 이어 붙임)
    sps = modem.samples_per_symbol(c)
    t = np.arange(sps) / c.fs
    return np.concatenate([np.sin(2 * np.pi * (c.f1 if b else c.f0) * t) for b in bits])


def step1():
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)
    x = modem.fsk_modulate(bits, cfg)
    t_ms = np.arange(len(x)) / cfg.fs * 1000
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5))
    ax[0].plot(t_ms, x, lw=0.8)
    for k in range(1, len(bits)):
        ax[0].axvline(k * SPS / cfg.fs * 1000, color="gray", ls="--", lw=0.8)
    for k, b in enumerate(bits):
        ax[0].text((k + 0.5) * SPS / cfg.fs * 1000, 1.1, f"bit {b}", ha="center")
    ax[0].set_ylim(-1.3, 1.4); ax[0].set_xlabel("time (ms)"); ax[0].set_title(f"BFSK waveform ({cfg.baud:g} baud)")
    peak_amp = float(np.abs(x).max()) if len(x) else 0.0
    if not 0.9 < peak_amp < 1.1:
        print(f"[주의] 파형의 최대 크기가 {peak_amp:.3g} 입니다. sin 파형이면 -1 ~ 1 사이를 오가야 합니다 "
              "(2π 나 fs 를 빠뜨렸는지 확인).")
    ax[1].plot(t_ms[:120], x[:120], ".-", lw=0.8)
    ax[1].set_xlabel("time (ms)"); ax[1].set_title("zoom: first 120 samples (one dot = one sample)")

    # 350 baud: 한 비트에 사인파가 정수 개로 안 떨어진다 -> 위상을 안 이으면 경계에서 튐
    c = argparse.Namespace(**vars(cfg)); c.baud = 350
    sps = modem.samples_per_symbol(c)
    y = modem.fsk_modulate(np.array([0, 1], dtype=np.uint8), c)
    lo, hi = sps - 25, sps + 25
    jump = np.abs(np.diff(y[lo:hi])).max()
    limit = 2 * np.pi * max(c.f0, c.f1) / c.fs
    if np.abs(y).max() < 0.5:
        verdict, col = "amplitude too small - check 2π and fs", "C3"
    elif jump > limit + 1e-6:
        verdict, col = "JUMP! phase is not continuous", "C3"
    else:
        verdict, col = "smooth (continuous phase)", "C2"
    ax[2].plot(np.arange(lo, hi), y[lo:hi], ".-", lw=0.8, color=col)
    ax[2].axvline(sps - 0.5, color="gray", ls="--")
    ax[2].set_title("bit boundary zoom at 350 baud:  " + verdict)
    ax[2].set_xlabel("sample index")
    save(fig, "explore_step1_wave.png")

    # 스펙트럼: 0만 보낼 때 / 1만 보낼 때
    fig, ax = plt.subplots(figsize=(9, 3))
    for b, col in [(0, "C0"), (1, "C1")]:
        z = modem.fsk_modulate(np.full(20, b, dtype=np.uint8), cfg)
        f = np.fft.rfftfreq(len(z), 1 / cfg.fs)
        ax.plot(f / 1000, np.abs(np.fft.rfft(z)), col, label=f"all bits = {b}")
    ax.set_xlim(0, 8); ax.set_xlabel("frequency (kHz)"); ax.legend()
    ax.set_title("spectrum = how strong each frequency is")
    save(fig, "explore_step1_spectrum.png")

    # 직접 들어 보기: 0 1 0 1 ... 을 느리게(10 baud) 재생하면 '삐-뽀-삐-뽀'
    slow = argparse.Namespace(**vars(cfg)); slow.baud = 10
    z = 0.3 * modem.fsk_modulate(np.array([0, 1] * 4, dtype=np.uint8), slow)
    wavfile.write(os.path.join(OUT, "explore_step1.wav"), cfg.fs, (z * 32767).astype(np.int16))
    print("소리 저장:", os.path.join(OUT, "explore_step1.wav"), "(더블클릭해서 들어 보세요)")
    print("\n질문) 파형에서 bit 0 구간과 bit 1 구간의 차이는? 스펙트럼의 두 봉우리는 몇 kHz에 있나?")
    print("      맨 아래 그림(350 baud)이 빨간색 'JUMP!' 이면 위상이 끊긴 것 -> check_week2.py 1 의 연속 위상 FAIL")


def step2():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 48).astype(np.uint8)
    x = ref_tones(bits)
    fig, ax = plt.subplots(3, 1, figsize=(9, 7))
    for i, snr in enumerate([0, -13, -16]):
        noise = rng.normal(0, np.sqrt(0.5 / 10 ** (snr / 10)), len(x))
        rx = modem.fsk_demodulate(x + noise, 0, len(bits), cfg)
        err = rx != bits
        ax[i].step(range(len(bits)), bits + 1.2, where="mid", label="sent (+1.2)")
        ax[i].step(range(len(bits)), rx, where="mid", label="received")
        ax[i].plot(np.flatnonzero(err), rx[err], "rx", ms=10, label="error")
        ax[i].set_title(f"SNR {snr} dB: {err.sum()} errors / {len(bits)} bits")
        ax[i].set_yticks([]); ax[i].legend(loc="upper right", fontsize=7)
    save(fig, "explore_step2_demod.png")

    print("SNR 별 BER (비트 2000개로 측정):")
    many = rng.integers(0, 2, 2000).astype(np.uint8)
    xm = ref_tones(many)
    for snr in [0, -8, -10, -12, -14, -16]:
        noise = rng.normal(0, np.sqrt(0.5 / 10 ** (snr / 10)), len(xm))
        ber = (modem.fsk_demodulate(xm + noise, 0, len(many), cfg) != many).mean()
        print(f"  SNR {snr:4d} dB  ->  BER {ber:.4f}")

    # 심볼 하나(비트 0 / 비트 1)의 스펙트럼 -> 어느 쪽 봉우리가 큰지가 곧 판정
    fig, ax = plt.subplots(figsize=(9, 3))
    for b, col in [(0, "C0"), (1, "C1")]:
        sym = ref_tones([b]) + rng.normal(0, 0.7, SPS)
        f = np.fft.rfftfreq(SPS, 1 / cfg.fs)
        ax.plot(f / 1000, np.abs(np.fft.rfft(sym)), col, label=f"one noisy symbol, bit {b}")
    for f0 in [cfg.f0, cfg.f1]:
        ax.axvline(f0 / 1000, color="gray", ls="--")
    ax.set_xlim(0, 8); ax.set_xlabel("frequency (kHz)"); ax.legend()
    save(fig, "explore_step2_symbol.png")
    print("\n참고) SNR 은 '샘플 하나' 기준이다. -8 dB 면 샘플 하나만 보면 잡음이 더 크지만,")
    print("      한 비트는 240개 샘플이고 그걸 전부 모아서(곱해서 더해서) 판단하기 때문에 잡음이 서로 상쇄된다.")
    print("질문) SNR 이 얼마일 때부터 에러가 생기나? 점선(3, 5 kHz) 두 곳의 높이만 비교하면 되는 이유는?")


def step3():
    # 프레임 조립(TODO 5) 없이: [무음 | chirp | guard | 비트 368개 | 무음] 을 두 번 직접 이어 붙인다
    rng = np.random.default_rng(2)
    chirp = modem.make_chirp(cfg)
    guard = np.zeros(int(cfg.guard * cfg.fs))
    parts = [np.zeros(int(0.5 * cfg.fs))]
    for _ in range(2):
        parts += [chirp, guard, modem.fsk_modulate(rng.integers(0, 2, 368).astype(np.uint8), cfg),
                  np.zeros(int(0.3 * cfg.fs))]
    sig = cfg.amp * np.concatenate(parts)
    rx = modem.simulate_channel(sig, cfg.fs, snr_db=10)
    ncorr = modem.normalized_xcorr(rx, chirp)
    peaks, pos = [], 0
    while True:
        p = modem.find_next_preamble(ncorr, pos, cfg.sync_thr, len(chirp))
        if p is None or len(peaks) > 10:
            break
        peaks.append(p); pos = p + len(chirp)
    t = np.arange(len(rx)) / cfg.fs
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.5))
    ax[0].specgram(np.concatenate([np.zeros(1000), chirp, np.zeros(1000)]) + 1e-4 * rng.normal(size=len(chirp) + 2000),
                   NFFT=256, Fs=cfg.fs, noverlap=224, cmap="magma")
    ax[0].set_ylim(0, 10000); ax[0].set_ylabel("Hz")
    ax[0].set_title("chirp = pitch sweeps 1 -> 8 kHz in 0.1 s (spectrogram: time vs frequency)")
    ax[1].plot(t, rx, lw=0.3); ax[1].set_title("received signal (2 packets, SNR 10 dB)")
    W = 200                                       # 그림에서 뾰족한 봉우리가 보이도록 구간별 최댓값
    ncm = ncorr[:len(ncorr) // W * W].reshape(-1, W).max(axis=1)
    ax[2].plot(t[:len(ncm) * W:W], ncm, lw=0.8, label="normalized correlation (similarity to chirp)")
    ax[2].axhline(cfg.sync_thr, color="gray", ls="--", label=f"threshold {cfg.sync_thr}")
    ax[2].plot(t[peaks], ncorr[peaks], "rv", ms=9, label="found by find_next_preamble")
    ax[2].set_ylim(0, 1.05); ax[2].set_xlabel("time (s)"); ax[2].legend(fontsize=8)
    for a in ax[1:]:
        for p in peaks:
            a.axvline(t[p], color="r", lw=0.6)
    save(fig, "explore_step3_sync.png" if cfg.sync_thr == 0.25 else f"explore_step3_sync_thr{cfg.sync_thr:g}.png")
    more = "  (10개 넘게 찾아서 멈춤 — 실제로는 더 많을 수 있음)" if len(peaks) > 10 else ""
    print(f"찾은 프레임 시작: {len(peaks)}개 (보낸 것 2개){more}")
    print("\n질문) 데이터(3, 5 kHz 톤) 구간에서는 왜 상관값이 작을까?")
    print("      임계값을 낮추면?  python explore_week2.py 3 --sync_thr 0.05")


def step4():
    header = modem.pack_header(3, 8, 256, 4)
    rep = np.tile(header, modem.HEADER_REP)
    bad = [2, 40, 75]                        # 사본마다 한 비트씩 에러 (사본1의 2번, 사본2의 8번, 사본3의 11번)
    rep[bad] ^= 1
    fixed = modem.majority_header(rep)
    show = lambda a: " ".join(map(str, a[:16]))
    print("원래 헤더 (앞 16비트):", show(header))
    for k in range(modem.HEADER_REP):
        marks = [" "] * 16
        for b in bad:
            if k * 32 <= b < k * 32 + 16:
                marks[b - k * 32] = "^"
        print(f"받은 사본 {k + 1}        :", show(rep[k * 32:(k + 1) * 32]))
        print("                      ", " ".join(marks), " <- 틀린 비트" if "^" in marks else "")
    print("다수결 결과          :", show(fixed), "-> 원래와 같음" if np.array_equal(fixed, header) else "-> 다름!")
    print("해석:", modem.unpack_header(fixed))
    print("\n질문) 같은 위치의 비트가 두 사본에서 틀리면? 3번 반복 대신 5번 반복하면 무엇이 좋아지고 무엇을 잃나?")


def step5():
    payload = np.random.default_rng(3).integers(0, 2, cfg.frame_bits).astype(np.uint8)
    fb = modem.build_frame_bits(0, 8, payload, 4)
    n_hdr = modem.HEADER_BITS * modem.HEADER_REP
    print(f"프레임 비트 수 {len(fb)} = 헤더 {n_hdr} + payload {len(payload)} + CRC {modem.CRC_BITS}")
    print(f"  비트 중 payload 비율: {len(payload) / len(fb):.0%}")
    dur = modem.frame_waveform(fb, cfg).size / cfg.fs
    t_payload = len(payload) / cfg.baud
    print(f"프레임 하나를 소리로 보내는 시간: {dur:.2f} s  (chirp, 헤더, CRC, 무음 포함)")
    print(f"  그중 payload 를 보내는 시간: {t_payload:.2f} s  ->  시간 효율 {t_payload / dur:.0%}")
    good = np.array_equal(fb[-16:], modem.crc16(np.concatenate([modem.pack_header(0, 8, len(payload), 4), payload])))
    print("내 프레임의 마지막 16비트 = crc16(헤더 1개 + payload) ?", "맞음" if good else "틀림! (TODO 5 확인)")
    bad = payload.copy(); bad[10] ^= 1
    ok = np.array_equal(modem.crc16(np.concatenate([fb[:32], bad])), fb[-16:])
    print("payload 1비트가 틀리면 CRC 일치?", ok, "(False 여야 에러를 알아챈 것)")
    print("\n질문) 프레임을 짧게/길게 하면 효율과 프레임 손실 위험은 어떻게 변할까?")
    print("      python explore_week2.py 5 --frame_bits 64   /   --frame_bits 1024 로 비교해 보자")


STEPS = {1: step1, 2: step2, 3: step3, 4: step4, 5: step5}

if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].isdigit() or int(sys.argv[1]) not in STEPS:
        print(__doc__)
        sys.exit()
    n = int(sys.argv[1])
    try:
        STEPS[n]()
    except NotImplementedError as e:
        if str(e).startswith("TODO"):
            print(f"아직 안 채운 함수가 있어요: {e}  -> modem.py 의 그 TODO를 먼저 채우세요")
        else:
            print(f"실행 중 에러: NotImplementedError: {e}")
    except Exception as e:
        print(f"실행 중 에러: {type(e).__name__}: {e}")
        print(f"-> 먼저  python check_week2.py {n}  를 통과했는지 확인하세요 (내 코드가 아직 틀렸을 수 있음)")
