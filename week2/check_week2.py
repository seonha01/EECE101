"""
Week 2 자가 채점: modem.py 빈칸이 맞게 동작하는지 작은 예제로 확인한다.

    python check_week2.py        # 전체
    python check_week2.py 1      # 단계 1만  (1~5, 6 = 전체 연결)
    python check_week2.py 1 2    # 단계 1, 2

모든 항목이 [PASS]면 loopback.py -> tx.py/rx.py 순서로 진행.
"""
import sys
import argparse
import numpy as np
import modem

cfg = modem.add_modem_args(argparse.ArgumentParser()).parse_args([])
rng = np.random.default_rng(0)
steps = {int(a) for a in sys.argv[1:]}        # 비어 있으면 전체
results = []


def check(step, name, fn):
    # fn()이 True면 PASS. 아직 안 채운 TODO면 [TODO]로 표시
    if steps and step not in steps:
        return
    try:
        ok = bool(fn())
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] 단계{step}  {name}")
    except NotImplementedError as e:
        results.append(False)
        if str(e).startswith("TODO"):           # 우리가 심어 둔 빈칸
            print(f"[TODO] 단계{step}  {name}  <- {e}  (코드를 썼다면 raise 줄을 지웠는지 확인)")
        else:                                   # numpy 등이 낸 에러 -> 내 코드 문제
            print(f"[ERR ] 단계{step}  {name}  <- NotImplementedError: {e}")
    except Exception as e:
        results.append(False)
        print(f"[ERR ] 단계{step}  {name}  <- {type(e).__name__}: {e}")


sps = modem.samples_per_symbol(cfg)
bits = rng.integers(0, 2, 64).astype(np.uint8)

# ---------------- 단계 1) BFSK 변조 ----------------
check(1, "fsk_modulate: 길이 = 비트 수 x 심볼당 샘플 수",
      lambda: len(modem.fsk_modulate(bits, cfg)) == len(bits) * sps)


def tone_ok():
    # 비트 0만 보내면 3 kHz, 비트 1만 보내면 5 kHz 인지 FFT로 확인
    for b, f in [(0, cfg.f0), (1, cfg.f1)]:
        x = modem.fsk_modulate(np.array([b] * 10, dtype=np.uint8), cfg)
        peak = np.argmax(np.abs(np.fft.rfft(x))) * cfg.fs / len(x)
        if abs(peak - f) > 50:
            return False
    return True


def smooth_ok():
    # 위상이 이어지는지: 이웃 샘플 사이 변화량이 한 번도 크게 튀지 않아야 함
    # (350 baud: 한 비트에 사인파가 정수 개로 딱 떨어지지 않아서, 위상을 안 이으면 경계에서 튄다)
    c = argparse.Namespace(**vars(cfg)); c.baud = 350
    x = modem.fsk_modulate(np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8), c)
    max_step = 2 * np.pi * max(c.f0, c.f1) / c.fs           # 사인파가 샘플 하나에 변할 수 있는 최대치
    return np.abs(np.diff(x)).max() <= max_step + 1e-6


check(1, "fsk_modulate: 0 -> f0, 1 -> f1 주파수", tone_ok)
check(1, "fsk_modulate: 비트가 바뀌어도 파형이 끊기지 않음 (연속 위상, --baud 350 으로 검사)", smooth_ok)


# ---------------- 단계 2) BFSK 복조 ----------------
def ref_tones(bs):
    # 채점용 송신 신호: 비트마다 사인파를 따로 만들어 이어 붙임 (200 baud 에서는 이 방법도 끊기지 않는다)
    t = np.arange(sps) / cfg.fs
    return np.concatenate([np.sin(2 * np.pi * (cfg.f1 if b else cfg.f0) * t) for b in bs])


x_ref = ref_tones(bits)
check(2, "fsk_demodulate: 반환 길이 = 비트 수  (행마다 따로 계산했는지, axis 확인)",
      lambda: len(np.atleast_1d(modem.fsk_demodulate(x_ref, 0, len(bits), cfg))) == len(bits))
check(2, "fsk_demodulate: 잡음 없는 신호 -> 원래 비트",
      lambda: np.array_equal(modem.fsk_demodulate(x_ref, 0, len(bits), cfg), bits))
check(2, "fsk_demodulate: 잡음(SNR 0 dB)에서도 대부분 맞음",
      lambda: (modem.fsk_demodulate(x_ref + rng.normal(0, np.sqrt(0.5), len(x_ref)), 0, len(bits), cfg)
               != bits).mean() < 0.05)
check(2, "fsk_demodulate: 음량이 작아도 (x0.01) 맞음",
      lambda: np.array_equal(modem.fsk_demodulate(0.01 * x_ref, 0, len(bits), cfg), bits))

# ---------------- 단계 3) 동기화 ----------------
ncorr = np.array([0.0, 0.1, 0.3, 0.5, 0.9, 0.6, 0.2, 0.0, 0.0, 0.8, 0.1])
check(3, "find_next_preamble: 임계값 넘은 첫 구간의 최대 피크 (=4)",
      lambda: modem.find_next_preamble(ncorr, 0, 0.25, 4) == 4)
check(3, "find_next_preamble: pos 이후에서 찾기, 전체 기준 위치 (=9)",
      lambda: modem.find_next_preamble(ncorr, 7, 0.25, 4) == 9)
check(3, "find_next_preamble: 없으면 None",
      lambda: modem.find_next_preamble(ncorr, 10, 0.25, 4) is None)

# ---------------- 단계 4) 헤더 다수결 ----------------
header = modem.pack_header(3, 8, 256, 4)
rep = np.tile(header, modem.HEADER_REP)
rep[5] ^= 1                                  # 첫 번째 사본의 5번 비트 에러 (0 -> 1)
rep[modem.HEADER_BITS + 20] ^= 1             # 두 번째 사본의 20번 비트 에러 (0 -> 1)
rep[2 * modem.HEADER_BITS + 7] ^= 1          # 세 번째 사본의 7번 비트 에러 (1 -> 0)
check(4, "majority_header: 에러 없으면 그대로  (반환 길이 32 = 열마다 다수결했는지, axis 확인)",
      lambda: np.array_equal(modem.majority_header(np.tile(header, modem.HEADER_REP)), header))
check(4, "majority_header: 사본 하나씩 틀려도 복원",
      lambda: np.array_equal(modem.majority_header(rep), header))

# ---------------- 단계 5) 프레임 조립 ----------------
payload = rng.integers(0, 2, 256).astype(np.uint8)
n_hdr = modem.HEADER_BITS * modem.HEADER_REP


def frame(part):
    fb = modem.build_frame_bits(3, 8, payload, 4)
    if part == "len":
        return len(fb) == n_hdr + 256 + 16
    if part == "header":
        return np.array_equal(fb[:n_hdr], np.tile(header, modem.HEADER_REP))
    if part == "payload":
        return np.array_equal(fb[n_hdr:n_hdr + 256], payload)
    return np.array_equal(fb[-16:], modem.crc16(np.concatenate([header, payload])))


check(5, "build_frame_bits: 길이 = 96 + 256 + 16", lambda: frame("len"))
check(5, "build_frame_bits: 앞 96비트 = 헤더 통째로 3번 [h|h|h]  (원소마다 반복 h0 h0 h0 h1... 이 아님)",
      lambda: frame("header"))
check(5, "build_frame_bits: 그다음 = payload", lambda: frame("payload"))
check(5, "build_frame_bits: 마지막 16비트 = CRC(header 1개 + payload)", lambda: frame("crc"))

# ---------------- 단계 6) 전체 연결 (채널 시뮬레이션) ----------------
data = rng.integers(0, 2, 1024).astype(np.uint8)


def end_to_end(snr):
    sig, _ = modem.build_tx_signal(data, 4, cfg)
    frames = modem.decode_frames(modem.simulate_channel(sig, cfg.fs, snr_db=snr), cfg)
    rx, valid = modem.assemble_bits(frames, len(data), cfg.frame_bits)
    return valid.all() and (rx != data).mean() < 1e-3


E2E = [("loopback SNR 10 dB: 모든 프레임 수신, BER < 1e-3", 10),
       ("loopback SNR 0 dB: 모든 프레임 수신, BER < 1e-3", 0)]
if not steps and not all(results):
    # 단계 1~5 에 실패가 있으면 전체 연결은 의미가 없고, 잘못된 TODO 3 은 끝나지 않을 수도 있다
    for name, _ in E2E:
        results.append(False)
        print(f"[SKIP] 단계6  {name}  <- 단계 1~5 를 먼저 모두 통과하세요")
else:
    for name, snr in E2E:
        check(6, name, lambda s=snr: end_to_end(s))

print(f"\n{sum(results)}/{len(results)} 통과")
if results and all(results):
    print("모두 통과!" + ("" if steps else "  이제 README 단계 6 (loopback) 으로 넘어가세요."))
