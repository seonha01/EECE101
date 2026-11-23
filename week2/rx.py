"""
수신기 (노트북 B): 마이크 / wav -> 프레임 검출 -> BFSK 복조 -> 비트열 -> Decoder -> 이미지

예)  python rx.py --wav tx.wav                   # wav 복조
     python rx.py --record 30 --save_wav rx.wav   # 30초 녹음 후 복조

BER 계산용 정답 비트는 같은 모델/같은 이미지(--index)를 수신 측에서 다시 인코딩해 얻는다.
"""
import os
import argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modem
import image_codec as codec
import utils


def read_wav(path, fs):
    sr, sig = wavfile.read(path)
    if sig.dtype.kind == 'i':
        sig = sig / np.iinfo(sig.dtype).max
    if sig.ndim > 1:
        sig = sig.mean(axis=1)                      # 스테레오 -> 모노
    if sr != fs:
        sig = resample_poly(sig, fs, sr)
    return sig.astype(np.float64)


def record(seconds, fs, device=None):
    import sounddevice as sd
    print(f"{seconds}초 녹음 시작... 이제 송신기를 재생하세요")
    sig = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32', device=device)
    sd.wait()
    print("녹음 완료")
    return sig[:, 0].astype(np.float64)


def receive(sig, model, x_true, cfg, quiet=False):
    """수신 신호 -> (결과 dict, 복원 이미지, 채널 없는 복원 이미지). quiet=False 면 문제 진단도 출력"""
    true_bits, shape = codec.image_to_bits(model, x_true, cfg.gray, cfg.interleave)
    n_frames = int(np.ceil(len(true_bits) / cfg.frame_bits))

    frames = modem.decode_frames(sig, cfg)
    bits, valid = modem.assemble_bits(frames, len(true_bits), cfg.frame_bits)

    x_rx = codec.bits_to_image(model, bits, shape, valid, cfg.gray, cfg.interleave)
    x_clean = codec.bits_to_image(model, true_bits, shape, None, cfg.gray, cfg.interleave)
    n_rx = valid.sum()
    rep = dict(
        frames_total=n_frames,
        frames_rx=len({f['seq'] for f in frames if f['seq'] < n_frames}),
        crc_fail=sum(not f['crc_ok'] for f in frames),
        ber=float((bits[valid] != true_bits[valid]).mean()) if n_rx else float('nan'),
        psnr_rx=codec.psnr(x_rx, x_true),
        psnr_clean=codec.psnr(x_clean, x_true),
    )
    rep['frame_loss'] = 1 - rep['frames_rx'] / n_frames
    rep['chirps'], rep['best_corr'] = count_chirps(sig, cfg)
    rep['tips'] = [] if quiet else diagnose(sig, frames, rep, cfg)
    return rep, x_rx, x_clean


def count_chirps(sig, cfg):
    # 헤더를 읽었는지와 상관없이, chirp(프레임 시작 소리)를 몇 번 찾았나
    chirp = modem.make_chirp(cfg)
    ncorr = modem.normalized_xcorr(sig, chirp)
    n, pos = 0, 0
    while True:
        p = modem.find_next_preamble(ncorr, pos, cfg.sync_thr, len(chirp))
        if p is None or p < pos:                      # p < pos 이면 TODO 3 오류 (무한 반복 방지)
            break
        n, pos = n + 1, p + len(chirp)
    return n, float(ncorr.max()) if len(ncorr) else 0.0


def diagnose(sig, frames, rep, cfg):
    """결과가 이상할 때 원인 후보를 알려 준다"""
    tips = []
    rms = float(np.sqrt(np.mean(sig ** 2)))
    n_chirp, best = rep['chirps'], rep['best_corr']
    if rms < 0.003:
        tips.append(f"녹음 소리가 매우 작습니다 (rms {rms:.4f}). 송신 쪽 볼륨을 올리고 노트북을 가까이 두세요.")
    if n_chirp == 0:
        tips.append(f"chirp(프레임 시작 소리)를 하나도 못 찾았습니다 (최대 유사도 {best:.2f} < 임계값 {cfg.sync_thr}). "
                    "소리가 마이크에 안 들어왔거나 너무 작습니다. --device 로 마이크를 바꿔 보거나 --sync_thr 를 낮춰 보세요.")
    elif rep['frames_rx'] < min(n_chirp, rep['frames_total']):
        tips.append(f"chirp 는 {n_chirp}번 찾았는데 프레임은 {rep['frames_rx']}개만 읽었습니다 = chirp 뒤의 소리(헤더)를 못 읽음. "
                    "(1) tx 와 rx 의 --baud, --f0, --f1, --frame_bits 가 같은지 확인하세요. "
                    "(2) 진짜 소리를 같은 노트북의 스피커 -> 마이크로 하고 있다면, 노트북 마이크의 '에코 제거' 기능이 "
                    "자기 스피커 소리를 지우는 경우가 많습니다. 노트북 두 대로 하거나, tx.wav 를 휴대폰으로 재생해 보세요.")
    if frames and rep['frames_rx'] < rep['frames_total']:
        frame_len = len(modem.frame_waveform(np.zeros(cfg.frame_bits + 112, np.uint8), cfg))
        if frames[-1]['peak'] + 2 * frame_len > len(sig):
            tips.append("녹음이 송신 도중에 끝난 것 같습니다. --record 를 tx.py 가 알려 준 시간보다 길게 주세요.")
    if rep['frames_rx'] > 0 and rep['crc_fail'] == 0 and rep['ber'] > 0.3:
        tips.append("CRC 는 모두 통과했는데 BER 이 0.5 근처입니다 = 전송은 성공, 비교할 정답이 다름. "
                    "tx 와 rx 의 --index, --interleave, --gray, --ckpt 가 같은지 확인하세요.")
    return tips


def print_report(rep):
    print(f"프레임 시작(chirp) 찾음 {rep.get('chirps', '?')}번  ->  헤더까지 읽은 프레임 "
          f"{rep['frames_rx']}/{rep['frames_total']}  (손실률 {rep['frame_loss']:.1%}),  CRC 실패 {rep['crc_fail']}개")
    if np.isnan(rep['ber']):
        print("BER: 받은 비트가 없어서 계산할 수 없음")
    else:
        print(f"BER (수신된 비트 기준) {rep['ber']:.2e}")
    print(f"PSNR  채널 없음 {rep['psnr_clean']:.2f} dB  ->  수신 이미지 {rep['psnr_rx']:.2f} dB")
    for t in rep.get('tips', []):
        print("[진단]", t)


def save_figure(x_true, x_clean, x_rx, rep, path):
    fig, ax = plt.subplots(1, 3, figsize=(7, 2.8))
    titles = ["Original", f"No channel\n{rep['psnr_clean']:.2f} dB",
              f"Received\n{rep['psnr_rx']:.2f} dB " + ("(no bits)" if np.isnan(rep['ber']) else f"(BER {rep['ber']:.1e})")]
    for a, img, t in zip(ax, [x_true, x_clean, x_rx], titles):
        a.imshow(codec.to_img(img)); a.set_title(t, fontsize=9); a.axis("off")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print("그림 저장:", path)


def main():
    parser = argparse.ArgumentParser()
    modem.add_modem_args(parser)
    codec.add_codec_args(parser)
    parser.add_argument("--wav", type=str, default=None, help="복조할 wav 파일")
    parser.add_argument("--record", type=float, default=None, help="마이크 녹음 시간 (s)")
    parser.add_argument("--save_wav", type=str, default=None, help="녹음을 wav로 저장")
    parser.add_argument("--device", default=None, help="sounddevice 입력 장치 번호/이름")
    parser.add_argument("--fig", type=str, default=os.path.join(codec.RESULT_DIR, "rx_result.png"))
    args = parser.parse_args()

    model, _ = codec.load_model(args)
    x_true = codec.load_image(args)

    if args.wav:
        sig = read_wav(args.wav, args.fs)
    elif args.record:
        sig = record(args.record, args.fs, args.device)
        if args.save_wav:
            wavfile.write(args.save_wav, args.fs, (sig * 32767).astype(np.int16))
    else:
        parser.error("--wav 또는 --record 를 지정하세요")

    rep, x_rx, x_clean = receive(sig, model, x_true, args)
    print_report(rep)
    save_figure(x_true, x_clean, x_rx, rep, args.fig)


if __name__ == "__main__":
    main()
