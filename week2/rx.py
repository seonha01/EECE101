"""
수신기 (노트북 B): 마이크 / wav -> 프레임 검출 -> BFSK 복조 -> 비트열 -> Decoder -> 이미지

예)  python rx.py --wav tx.wav                   # wav 복조
     python rx.py --record 30 --save_wav rx.wav   # 30초 녹음 후 복조

BER 계산용 정답 비트는 같은 모델/같은 이미지(--index)를 수신 측에서 다시 인코딩해 얻는다.
"""
import os
import sys
import subprocess
import argparse
import numpy as np
import torch
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
    if isinstance(device, str) and device.isdigit():
        device = int(device)  # "--device 1" 은 장치 번호로 해석
    print(f"{seconds}초 녹음 시작... 이제 송신기를 재생하세요")
    sig = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32', device=device)
    sd.wait()
    print("녹음 완료")
    return sig[:, 0].astype(np.float64)


def receive(sig, model, x_true, cfg, quiet=False):
    """수신 신호 -> (결과 dict, 복원 이미지, 채널 없는 복원 이미지). quiet=False 면 문제 진단도 출력
    x_true=None 이면 원본 없이 받기만 한다 (BER/PSNR 생략)"""
    gh, gw = getattr(cfg, "grid", None) or (cfg.tiles, cfg.tiles)
    ref = x_true if x_true is not None else torch.zeros(1, 3, 32 * gh, 32 * gw)   # 비트 수/모양만 알면 됨
    true_bits, shape = codec.image_to_bits(model, ref, cfg.gray, cfg.interleave)   # 사진 부분 (정보 비트 제외)
    n_all = codec.INFO_LEN + len(true_bits)
    n_frames = int(np.ceil(n_all / cfg.frame_bits))

    frames = modem.decode_frames(sig, cfg)
    bits, valid = modem.assemble_bits(frames, n_all, cfg.frame_bits)
    bits, valid = bits[codec.INFO_LEN:], valid[codec.INFO_LEN:]                   # 사진 부분만

    x_rx = codec.bits_to_image(model, bits, shape, valid, cfg.gray, cfg.interleave, (gh, gw))
    x_clean = codec.bits_to_image(model, true_bits, shape, None, cfg.gray, cfg.interleave, (gh, gw)) \
        if x_true is not None else None
    n_rx = valid.sum() if x_true is not None else 0
    rep = dict(
        frames_total=n_frames,
        frames_rx=len({f['seq'] for f in frames if f['seq'] < n_frames}),
        crc_fail=sum(not f['crc_ok'] for f in frames),
        ber=float((bits[valid] != true_bits[valid]).mean()) if n_rx else float('nan'),
        psnr_rx=codec.psnr(x_rx, x_true) if x_true is not None else float('nan'),
        psnr_clean=codec.psnr(x_clean, x_true) if x_true is not None else float('nan'),
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
    if np.isnan(rep['psnr_clean']):
        print("원본 이미지를 안 줘서 BER/PSNR 은 생략 (비교하려면 --image 또는 --index)")
        return print_tips(rep)
    if np.isnan(rep['ber']):
        print("BER: 받은 비트가 없어서 계산할 수 없음")
    else:
        print(f"BER (수신된 비트 기준) {rep['ber']:.2e}")
    print(f"PSNR  채널 없음 {rep['psnr_clean']:.2f} dB  ->  수신 이미지 {rep['psnr_rx']:.2f} dB")
    print_tips(rep)


def print_tips(rep):
    for t in rep.get('tips', []):
        print("[진단]", t)


def save_figure(x_true, x_clean, x_rx, rep, path, show=False):
    """원본이 있으면 [원본 | 채널 없음 | 수신] 비교, 없으면 받은 이미지만. show=True 면 그림을 바로 연다"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if x_true is None:
        fig, a = plt.subplots(figsize=(4, 4))
        a.imshow(codec.to_img(x_rx)); a.axis("off")
        fig.tight_layout(pad=0)
    else:
        fig, ax = plt.subplots(1, 3, figsize=(7, 2.8))
        titles = ["Original", f"No channel\n{rep['psnr_clean']:.2f} dB",
                  f"Received\n{rep['psnr_rx']:.2f} dB " + ("(no bits)" if np.isnan(rep['ber']) else f"(BER {rep['ber']:.1e})")]
        for a, img, t in zip(ax, [x_true, x_clean, x_rx], titles):
            a.imshow(codec.to_img(img)); a.set_title(t, fontsize=9); a.axis("off")
        fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print("그림 저장:", path)
    if not show:
        return
    if sys.platform == "darwin":
        subprocess.run(["open", path])
    elif sys.platform == "win32":
        os.startfile(path)


def tx_duration(model, cfg):
    """같은 옵션(B, tiles, baud ...)으로 보내는 소리의 길이 (s)"""
    s = 32 * cfg.tiles                                     # 정사각형 사진 = 가장 긴 경우
    n_bits = codec.INFO_LEN + len(codec.image_to_bits(model, torch.zeros(1, 3, s, s), cfg.gray, cfg.interleave)[0])
    sig, _ = modem.build_tx_signal(np.zeros(n_bits, np.uint8), cfg.B, cfg)
    return len(sig) / cfg.fs


def match_sender(sig, args):
    """소리에서 보내는 쪽 설정을 읽어 맞춘다: 헤더의 B, 첫 프레임 맨 앞의 사진 정보(지문, 조각 배치).
    args.B, args.grid, args.sent_id 를 정하고, B 가 바뀌었으면 True (모델을 다시 불러야 함)"""
    args.grid, args.sent_id = None, None
    frames = modem.decode_frames(sig, args)
    if not frames:
        args.grid = (args.tiles, args.tiles)
        return False
    B = int(np.bincount([f['B'] for f in frames]).argmax())            # 여러 프레임의 다수결
    total = int(np.bincount([f['total'] for f in frames]).argmax())
    first = [f for f in frames if f['seq'] == 0 and len(f['payload']) >= codec.INFO_LEN]
    if first:
        img_id, (gh, gw) = codec.parse_info(first[0]['payload'])
        need = int(np.ceil((codec.INFO_LEN + 512 * B * gh * gw) / args.frame_bits))
        if 1 <= gh <= 15 and 1 <= gw <= 15 and need == total:             # 프레임 수와 맞으면 믿는다
            args.grid, args.sent_id = (gh, gw), img_id
    if args.grid is None:                                                # 첫 프레임을 못 읽음: 프레임 수로 추정
        t = max(1, int(round(((total * args.frame_bits - codec.INFO_LEN) / (512 * B)) ** 0.5)))
        args.grid = (t, t)
    changed = B in (2, 4, 8) and B != args.B
    if changed:
        print(f"[자동] 보내는 쪽은 B={B} 입니다 (받는 쪽 설정 B={args.B}). 맞춰서 복원합니다.")
        args.B, args.ckpt = B, None
    return changed


def main():
    parser = argparse.ArgumentParser()
    modem.add_modem_args(parser)
    codec.add_codec_args(parser)
    parser.add_argument("--wav", type=str, default=None, help="복조할 wav 파일")
    parser.add_argument("--record", type=float, default=None,
                        help="마이크 녹음 시간 (s). 안 주면 옵션으로 계산한 송신 길이 + 12초")
    parser.add_argument("--save_wav", type=str, default=None, help="녹음 저장 위치 (기본 results/rx_record.wav)")
    parser.add_argument("--device", default=None, help="sounddevice 입력 장치 번호/이름")
    parser.add_argument("--fig", type=str, default=os.path.join(codec.RESULT_DIR, "rx_result.png"))
    parser.add_argument("--no_show", action="store_true", help="결과 그림을 자동으로 열지 않음")
    args = parser.parse_args()

    model, _ = codec.load_model(args)

    if args.wav:
        sig = read_wav(args.wav, args.fs)
    else:
        if not args.record:                                # 녹음 시간: 보내는 소리 길이 + 여유 12초
            args.record = float(np.ceil(tx_duration(model, args)) + 12)
        sig = record(args.record, args.fs, args.device)
        args.save_wav = args.save_wav or os.path.join(codec.RESULT_DIR, "rx_record.wav")
        os.makedirs(os.path.dirname(os.path.abspath(args.save_wav)), exist_ok=True)
        wavfile.write(args.save_wav, args.fs, (sig * 32767).astype(np.int16))
        print("녹음 저장:", args.save_wav, f" (다시 복조: python rx.py --wav {args.save_wav} + 같은 옵션)")

    if match_sender(sig, args):                            # 보내는 쪽 B 가 다르면 모델을 다시 불러옴
        model, _ = codec.load_model(args)
    gh, gw = args.grid
    print(f"[자동] 사진 크기 {32 * gw}x{32 * gh} (32x32 조각 가로 {gw} x 세로 {gh})")

    # 원본 사진: --image / --index 로 주거나, 없으면 보낸 사진의 지문으로 images 폴더에서 찾는다
    if not args.image and args.index is None and args.sent_id is not None:
        args.image = codec.find_image(args.sent_id)
        print(f"[자동] 보낸 사진: {os.path.basename(args.image)} (images 폴더에서 찾음) -> 원본과 비교합니다"
              if args.image else "[자동] 보낸 사진이 이 컴퓨터의 images 폴더에 없음 -> 받은 사진만 보여 줍니다")
    x_true = codec.load_image(args) if (args.image or args.index is not None) else None
    rep, x_rx, x_clean = receive(sig, model, x_true, args)
    print_report(rep)
    save_figure(x_true, x_clean, x_rx, rep, args.fig, show=not args.no_show)


if __name__ == "__main__":
    main()
