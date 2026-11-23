"""
송신/수신 공용 음향 모뎀 (BFSK + chirp 동기화 + CRC-16)

한 프레임의 오디오 구조:
  [chirp preamble | guard(무음) | header x3 | payload | CRC-16 | gap(무음)]

- header (32 bit) = 프레임 번호(8) + 전체 프레임 수(8) + payload 길이(12) + B(4)
  헤더가 깨지면 payload 위치를 모르므로 3번 반복해 보내고 다수결로 복원한다.
- CRC-16은 (header + payload)에 대해 계산. CRC가 틀려도 payload는 버리지 않고 사용
  (비트 에러가 있어도 디코더가 이미지를 복원하도록 = Week 1의 BSC 실험과 같은 상황)

★ TODO 채우는 법: `raise NotImplementedError(...)` 줄을 **지우고** 그 자리에 코드를 쓴다.
"""
import numpy as np
from scipy.signal import chirp as sp_chirp, fftconvolve
from scipy.signal.windows import tukey

HEADER_BITS = 32
HEADER_REP = 3
CRC_BITS = 16


# ======================= 파라미터 =======================
def add_modem_args(parser):
    g = parser.add_argument_group("modem")
    g.add_argument("--fs", type=int, default=48000, help="샘플링 주파수")
    g.add_argument("--f0", type=float, default=3000, help="비트 0 주파수 (Hz)")
    g.add_argument("--f1", type=float, default=5000, help="비트 1 주파수 (Hz)")
    g.add_argument("--baud", type=float, default=200, help="심볼 레이트 (bit/s)")
    g.add_argument("--chirp_f0", type=float, default=1000, help="chirp 시작 주파수")
    g.add_argument("--chirp_f1", type=float, default=8000, help="chirp 끝 주파수")
    g.add_argument("--chirp_dur", type=float, default=0.1, help="chirp 길이 (s)")
    g.add_argument("--guard", type=float, default=0.02, help="chirp 뒤 무음 (s)")
    g.add_argument("--gap", type=float, default=0.1, help="프레임 사이 무음 (s)")
    g.add_argument("--amp", type=float, default=0.5, help="송신 진폭 (0~1)")
    g.add_argument("--frame_bits", type=int, default=256, help="프레임당 payload 비트 수")
    g.add_argument("--sync_thr", type=float, default=0.25, help="정규화 상관 임계값 (0~1)")
    return parser


def samples_per_symbol(cfg):
    return int(round(cfg.fs / cfg.baud))


# ======================= 비트 유틸 =======================
def int_to_bits(value, n):
    return np.array([(value >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


def bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def crc16(bits):
    """CRC-16-CCITT (다항식 0x1021, 초기값 0xFFFF) -> 16비트 배열"""
    reg = 0xFFFF
    for b in bits:
        top = ((reg >> 15) & 1) ^ int(b)
        reg = (reg << 1) & 0xFFFF
        if top:
            reg ^= 0x1021
    return int_to_bits(reg, CRC_BITS)


# ======================= 1. 변조: 비트 -> 소리 (TODO 1) =======================
def make_chirp(cfg):
    # 선형 주파수 스윕 + 양끝을 부드럽게(tukey 창) -> 날카로운 상관 피크
    n = int(cfg.chirp_dur * cfg.fs)
    t = np.arange(n) / cfg.fs
    return sp_chirp(t, cfg.chirp_f0, cfg.chirp_dur, cfg.chirp_f1) * tukey(n, 0.2)


def fsk_modulate(bits, cfg):
    # 1) 샘플마다의 주파수: 비트 1 -> f1, 0 -> f0, 한 비트를 samples_per_symbol 개 샘플만큼 반복
    #    예) bits=[0,1], 심볼당 3샘플 -> freq=[3000,3000,3000, 5000,5000,5000]
    freq = np.repeat(np.where(bits > 0, cfg.f1, cfg.f0), samples_per_symbol(cfg))
    # ------------------------------------------------------------------------------
    # TODO 1. BFSK 변조 (README 단계 1)
    #
    #   목표: freq 대로 소리가 나는 사인파 만들기. 반환 길이 = len(freq)
    #
    #   생각해 볼 것:
    #   (a) 주파수가 f로 일정하면 사인파는 sin(2π f t). 여기서 2π f t 를 "위상"이라고 한다.
    #       위상은 1초에 2π f 씩, 즉 샘플 하나(1/fs 초)마다 2π f / fs 씩 늘어난다.
    #   (b) 그런데 우리는 주파수가 비트마다 바뀐다. 그래도 "샘플마다 위상이 2π * (그 샘플의 주파수) / fs
    #       만큼 늘어난다"는 규칙은 똑같다. 그러면 n번째 샘플의 위상은?
    #       -> 처음부터 n번째까지 늘어난 양을 전부 더한 것
    #   (c) 위상을 구했으면 출력은 sin(위상)
    #
    #   왜 sin(2π f t) 를 비트마다 따로 만들어 이어 붙이지 않을까?
    #   -> 한 비트 동안 사인파가 정수 번 딱 떨어지지 않으면(예: --baud 350) 비트가 바뀌는 순간
    #      파형이 뚝 끊겨서 스피커에서 '딱' 소리가 나고 다른 주파수로 에너지가 샌다.
    #      (b) 방식은 어떤 baud 에서도 위상이 이어진다 (continuous-phase FSK).
    #
    #   쓸 만한 도구: numpy의 "누적합" 함수, np.sin, np.pi, cfg.fs
    #   확인: python explore_week2.py 1  (파형과 스펙트럼 그림)
    # ------------------------------------------------------------------------------
    raise NotImplementedError("TODO 1 fsk_modulate")


# ======================= 2. 복조: 소리 -> 비트 (TODO 2) =======================
def fsk_demodulate(sig, start, n_bits, cfg, trim=0.1):
    """비동기 에너지 검출: 각 심볼 구간에서 f0, f1 성분 크기를 비교"""
    sps = samples_per_symbol(cfg)
    need = start + n_bits * sps
    if need > len(sig):
        sig = np.concatenate([sig, np.zeros(need - len(sig))])
    sym = sig[start:need].reshape(n_bits, sps)
    a, b = int(trim * sps), sps - int(trim * sps)       # 심볼 양끝(전이 구간)은 버림
    t = np.arange(a, b) / cfg.fs
    seg = sym[:, a:b]                                   # (n_bits, 샘플 수): 행 하나가 심볼 하나
    # ------------------------------------------------------------------------------
    # TODO 2. BFSK 복조 (README 단계 2)
    #
    #   seg 의 각 행은 비트 하나 동안 받은 소리. 이 소리가 f0(3 kHz)에 가까운지 f1(5 kHz)에 가까운지 판단.
    #
    #   "소리 x 에 주파수 f 성분이 얼마나 들어 있나?"를 재는 방법 (복소수 없이):
    #       C = Σ_n  x[n] · cos(2π f t[n])      <- x 와 cos 파형을 샘플마다 곱해서 전부 더함
    #       S = Σ_n  x[n] · sin(2π f t[n])      <- sin 파형으로도 똑같이
    #       크기(f) = √(C² + S²)
    #   - 왜 되나? x 가 주파수 f 의 사인파면 곱한 값이 계속 같은 부호(+)로 쌓여서 C 나 S 가 커진다.
    #     x 가 다른 주파수면 곱한 값이 +, - 를 오가다 서로 지워져서 거의 0 이 된다.
    #   - 왜 cos 와 sin 둘 다? 받은 소리가 언제 시작했는지(위상)를 모르기 때문. 위상에 따라 C 가 크거나
    #     S 가 크거나 하는데, √(C²+S²) 는 위상과 상관없이 같다  -> "비동기 검출"
    #   - (복소수를 안다면) 위 계산은  | Σ x[n] · e^(-j2πf t[n]) |  한 줄과 똑같다. 파이썬에서 j 는 1j.
    #
    #   할 일:
    #   (1) 모든 심볼(행)에 대해 f0 성분 크기 e0, f1 성분 크기 e1 을 구한다 (각각 길이 n_bits 인 배열)
    #   (2) e1 이 더 크면 1, 아니면 0.  반환은 길이 n_bits 인 np.uint8 배열
    #
    #   t 는 이미 위에 만들어 둠 (심볼 안에서의 시각, 초 단위. 모든 행이 같은 t 를 쓴다).
    #   가장 쉬운 방법: for 문으로 seg 의 행을 하나씩 꺼내 계산 -> 결과를 리스트에 모아 np.array 로.
    #   빠른 방법: seg * np.cos(...) 는 (n_bits, 샘플 수) 모양 -> 행마다 더하려면 np.sum(..., axis=1)
    #   쓸 만한 도구: np.cos, np.sin, np.sqrt, np.pi, np.sum(axis=...), .astype
    #   확인: python explore_week2.py 2  (잡음 속 비트 판정, 심볼 하나의 스펙트럼)
    # ------------------------------------------------------------------------------
    raise NotImplementedError("TODO 2 fsk_demodulate")


# ======================= 3. 동기화: 프레임 시작 찾기 (TODO 3) =======================
def normalized_xcorr(sig, ref):
    """정규화 상호상관 (0~1): 음량에 상관없이 chirp와 얼마나 닮았는지"""
    n = len(ref)
    num = fftconvolve(sig, ref[::-1], mode="valid")
    energy = fftconvolve(sig ** 2, np.ones(n), mode="valid")
    return np.abs(num) / (np.linalg.norm(ref) * np.sqrt(np.maximum(energy, 1e-12)) + 1e-12)


def find_next_preamble(ncorr, pos, thr, n_chirp):
    """pos 이후 처음으로 임계값을 넘는 지점 근처(chirp 길이 이내)의 최대 피크 위치"""
    # ------------------------------------------------------------------------------
    # TODO 3. 프레임 시작 찾기 (README 단계 3)
    #
    #   ncorr[i] = "i번째 샘플부터 chirp가 시작된다고 했을 때 얼마나 닮았나" (0~1)
    #   chirp가 실제로 있는 곳 근처에서만 값이 커지고, 정확한 시작점에서 가장 크다.
    #
    #   두 단계로 찾는다:
    #   (1) pos 부터 앞으로 보면서 처음으로 thr 를 넘는 위치 first 를 찾는다.
    #       -> 하나도 없으면 더 이상 프레임이 없는 것이므로 None 을 반환
    #   (2) 처음 넘은 곳이 꼭대기는 아니다 (값이 올라가는 중일 수 있음).
    #       first 부터 n_chirp 개 샘플 안에서 가장 큰 값이 있는 위치가 진짜 시작점.
    #
    #   예) ncorr = [0, .1, .3, .5, .9, .6, .2, 0, 0, .8, .1],  thr = 0.25
    #       pos=0 -> (1) first=2  (2) 2~5번 중 최대는 4번 -> 4 반환
    #       pos=7 -> (1) first=9  (2) -> 9 반환
    #       pos=10 -> 넘는 곳 없음 -> None
    #
    #   주의: ncorr[pos:] 처럼 잘라서 찾으면 위치가 pos 만큼 밀린다. 반환은 "전체 배열 기준" 위치(int).
    #   쓸 만한 도구: 조건을 만족하는 인덱스 찾기 (np.flatnonzero / np.where), 최댓값 위치 (np.argmax)
    #   확인: python explore_week2.py 3  (상관 그래프)
    # ------------------------------------------------------------------------------
    raise NotImplementedError("TODO 3 find_next_preamble")


# ======================= 4~5. 프레임: 헤더 다수결, 프레임 조립 (TODO 4, 5) =======================
def pack_header(seq, total, length, B):
    return np.concatenate([int_to_bits(seq, 8), int_to_bits(total, 8),
                           int_to_bits(length, 12), int_to_bits(B, 4)])


def unpack_header(h):
    return dict(seq=bits_to_int(h[0:8]), total=bits_to_int(h[8:16]),
                length=bits_to_int(h[16:28]), B=bits_to_int(h[28:32]))


def majority_header(rep_bits):
    # 3번 반복된 헤더를 비트별 다수결로 하나로 합침
    h = rep_bits.reshape(HEADER_REP, HEADER_BITS)          # (3, 32): 행 하나가 헤더 사본 하나
    # ------------------------------------------------------------------------------
    # TODO 4. 다수결 (README 단계 4)
    #
    #   h는 같은 헤더를 세 번 받은 것. 잡음 때문에 사본마다 몇 비트씩 틀려 있을 수 있다.
    #
    #             열0 열1 열2 ...
    #     사본1 [  1,  0,  1, ... ]
    #     사본2 [  1,  1,  1, ... ]     <- 열1이 한 번 틀림
    #     사본3 [  0,  0,  1, ... ]     <- 열0이 한 번 틀림
    #     결과  [  1,  0,  1, ... ]     <- 열마다 "더 많이 나온 값"
    #
    #   세 개 중 1이 몇 개인지 세면 0~3 중 하나. 몇 개 이상일 때 1로 정하면 될까?
    #   반환: 길이 32, 값이 0/1 인 np.uint8 배열
    #   쓸 만한 도구: 배열의 한 방향(axis)으로 더하기, 비교 연산(>=), .astype
    # ------------------------------------------------------------------------------
    raise NotImplementedError("TODO 4 majority_header")


def build_frame_bits(seq, total, payload, B):
    header = pack_header(seq, total, len(payload), B)
    # ------------------------------------------------------------------------------
    # TODO 5. 프레임 조립 (README 단계 5)
    #
    #   한 프레임의 비트 순서:
    #     [ header | header | header | payload | CRC(16비트) ]
    #       <------ HEADER_REP 번 ----->
    #
    #   - CRC는 "오류 검사용 요약값". 수신 측은 받은 header와 payload로 CRC를 다시 계산해서
    #     같이 받은 CRC와 비교한다 (-> decode_frames 의 crc_ok 부분을 읽어 보자).
    #     그러니 송신 측도 **반복하기 전의 header 한 개 + payload** 로 CRC를 계산해야 짝이 맞는다.
    #   - CRC 계산 함수는 위에 있는 crc16(bits) 를 쓰면 된다.
    #   반환: 1차원 0/1 배열.  길이 = 32*3 + len(payload) + 16
    #   쓸 만한 도구: 배열 이어 붙이기, 배열을 여러 번 반복하기 (numpy 문서에서 찾아보기)
    #   주의: fsk_modulate 에서 본 np.repeat 는 "원소 하나하나"를 반복한다:  [1, 0] -> [1, 1, 1, 0, 0, 0]
    #         여기서 필요한 건 "배열 통째로" 반복:                             [1, 0] -> [1, 0, 1, 0, 1, 0]
    # ------------------------------------------------------------------------------
    raise NotImplementedError("TODO 5 build_frame_bits")


def split_frames(bits, frame_bits):
    return [bits[i:i + frame_bits] for i in range(0, len(bits), frame_bits)]


def frame_waveform(frame_bits, cfg):
    guard = np.zeros(int(cfg.guard * cfg.fs))
    gap = np.zeros(int(cfg.gap * cfg.fs))
    return np.concatenate([make_chirp(cfg), guard, fsk_modulate(frame_bits, cfg), gap])


def build_tx_signal(bits, B, cfg, lead=0.5):
    """전체 비트열 -> 프레임 분할 -> 오디오 신호. (signal, 프레임 수) 반환"""
    chunks = split_frames(bits, cfg.frame_bits)
    silence = np.zeros(int(lead * cfg.fs))
    parts = [silence]
    for seq, payload in enumerate(chunks):
        fb = build_frame_bits(seq, len(chunks), payload, B)
        parts.append(frame_waveform(fb, cfg))
    parts.append(silence)
    return cfg.amp * np.concatenate(parts), len(chunks)


# ======================= 수신 전체: 프레임 찾기 -> 헤더 -> payload -> CRC =======================
def header_is_valid(h, cfg):
    return 0 < h['total'] <= 255 and h['seq'] < h['total'] and \
        0 < h['length'] <= cfg.frame_bits and 1 <= h['B'] <= 15


def decode_frames(sig, cfg):
    """수신 신호 전체에서 프레임들을 찾아 복조. 프레임 dict 리스트 반환"""
    sps = samples_per_symbol(cfg)
    chirp = make_chirp(cfg)
    ncorr = normalized_xcorr(sig, chirp)
    n_hdr = HEADER_BITS * HEADER_REP
    frames, pos = [], 0
    while True:
        peak = find_next_preamble(ncorr, pos, cfg.sync_thr, len(chirp))
        if peak is None:
            break
        if not pos <= peak < len(ncorr):              # 안전장치: 앞으로 안 나가면 무한 반복이 된다
            raise ValueError(f"find_next_preamble 이 pos={pos} 이후가 아닌 위치 {peak} 를 돌려줬습니다. "
                             "TODO 3 을 확인하세요 (pos 를 더했나? 없으면 None 을 돌려주나?)")
        data_start = peak + len(chirp) + int(cfg.guard * cfg.fs)
        h = unpack_header(majority_header(fsk_demodulate(sig, data_start, n_hdr, cfg)))
        if not header_is_valid(h, cfg):          # 잘못 잡은 피크 -> 조금 뒤부터 다시 탐색
            pos = peak + len(chirp)
            continue
        body_start = data_start + n_hdr * sps
        body = fsk_demodulate(sig, body_start, h['length'] + CRC_BITS, cfg)
        payload, crc = body[:h['length']], body[h['length']:]
        crc_ok = np.array_equal(crc16(np.concatenate([pack_header(**h), payload])), crc)
        frames.append(dict(h, payload=payload, crc_ok=crc_ok, peak=int(peak),
                           corr=float(ncorr[peak])))
        pos = body_start + len(body) * sps
    return frames


def assemble_bits(frames, n_bits, frame_bits):
    """프레임들을 원래 비트열 위치에 채워 넣음. (bits, 수신여부 mask) 반환"""
    bits = np.zeros(n_bits, dtype=np.uint8)
    valid = np.zeros(n_bits, dtype=bool)
    for f in frames:
        s = f['seq'] * frame_bits
        e = min(s + len(f['payload']), n_bits)
        if s >= n_bits or valid[s:e].all():       # 범위 밖이거나 중복 프레임은 무시
            continue
        bits[s:e] = f['payload'][:e - s]
        valid[s:e] = True
    return bits, valid


# ======================= 채널 시뮬레이션 =======================
def simulate_channel(x, fs, snr_db=20.0, delay=0.3, ppm=50.0, gain=0.3,
                     echo_gain=0.2, echo_delay=0.003, seed=0):
    """스피커->공기->마이크를 흉내: 감쇠, 지연, 에코, 클럭 오차(ppm), 백색잡음"""
    rng = np.random.default_rng(seed)
    d = int(echo_delay * fs)
    y = x + echo_gain * np.concatenate([np.zeros(d), x[:-d]])        # 반사파
    ratio = 1 + ppm * 1e-6                                           # 두 기기 클럭 차이
    t_new = np.arange(int(len(y) * ratio)) / ratio
    y = gain * np.interp(t_new, np.arange(len(y)), y)
    y = np.concatenate([np.zeros(int(delay * fs)), y, np.zeros(int(0.2 * fs))])
    p_sig = np.mean(y[np.abs(y) > 1e-6] ** 2)
    noise = rng.normal(0, np.sqrt(p_sig / 10 ** (snr_db / 10)), len(y))
    return y + noise
