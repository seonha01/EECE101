"""
수신기 (노트북 B): 마이크 / wav -> 프레임 검출 -> BFSK 복조 -> 비트열 -> Decoder -> 이미지

예)  python rx.py                      # 녹음 (시간은 옵션으로 계산) -> 복원 -> results/rx_result.png
     python rx.py --wav tx.wav         # wav 파일 복원

B, 사진 크기(조각 배치), 보낸 사진은 소리에서 알아낸다. 보낸 사진이 images 폴더에 있으면
그 사진을 다시 인코딩해서 정답 비트로 쓰고 BER / PSNR 을 계산한다.
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
    n_chirp, best, total = rep['chirps'], rep['best_corr'], rep['frames_total']
    got = {f['seq'] for f in frames if f['seq'] < total}
    frame_len = len(modem.frame_waveform(np.zeros(cfg.frame_bits + 112, np.uint8), cfg))
    cut_end = bool(got) and max(got) < total - 1 and frames[-1]['peak'] + 2 * frame_len > len(sig)

    if rms < 0.003:
        tips.append(f"녹음 소리가 매우 작습니다 (rms {rms:.4f}). 송신 쪽 볼륨을 올리고 노트북을 가까이 두세요.")
    if n_chirp == 0:
        tips.append(f"chirp(프레임 시작 소리)를 하나도 못 찾았습니다 (최대 유사도 {best:.2f} < 임계값 {cfg.sync_thr}). "
                    "소리가 마이크에 안 들어왔거나 너무 작습니다. --device 로 마이크를 바꿔 보거나 --sync_thr 를 낮춰 보세요.")
    elif min(n_chirp, total) - rep['frames_rx'] > int(cut_end):      # 끝에서 잘린 1개는 아래에서 따로 안내
        tips.append(f"chirp 는 {n_chirp}번 찾았는데 프레임은 {rep['frames_rx']}개만 읽었습니다 = chirp 뒤의 소리(헤더)를 못 읽음. "
                    "(1) 양쪽 --baud, --f0, --f1, --frame_bits 가 같은지 확인하세요. "
                    "(2) 한 노트북의 스피커 -> 같은 노트북 마이크라면, 마이크의 '에코 제거'가 자기 소리를 지웁니다. 노트북 두 대로 하세요. "
                    "(3) 방이 울리면 더 가깝게, 또는 --baud 를 낮추세요.")
    if got and 0 not in got:
        tips.append(f"앞부분 프레임 {min(got)}개를 놓쳤습니다 = 녹음을 늦게 시작함. 받는 쪽을 먼저 시작하고 '녹음 시작'을 본 뒤 보내세요.")
    if cut_end:
        tips.append("뒷부분을 놓쳤습니다 = 녹음이 송신 도중에 끝남. 받는 쪽에 보내는 쪽과 같은 B / --tiles 를 주거나 --record 를 늘리세요.")
    if rep['frames_rx'] > 0 and rep['crc_fail'] == 0 and rep['ber'] > 0.3:
        tips.append("CRC 는 모두 통과했는데 BER 이 0.5 근처입니다 = 전송은 성공, 비교한 원본이 보낸 사진과 다름. "
                    "--image / --index 를 빼서 자동으로 찾게 하거나, --interleave / --gray 가 양쪽 같은지 확인하세요.")
    return tips


def print_report(rep):
    print(f"프레임 시작(chirp) 찾음 {rep.get('chirps', '?')}번  ->  헤더까지 읽은 프레임 "
          f"{rep['frames_rx']}/{rep['frames_total']}  (손실률 {rep['frame_loss']:.1%}),  CRC 실패 {rep['crc_fail']}개")
    if np.isnan(rep['psnr_clean']):
        print("비교할 원본이 없어서 BER/PSNR 은 생략 (보낸 사진을 이 컴퓨터 images 폴더에도 넣으면 자동 비교)")
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


def get_signal(args, model):
    """--wav 파일을 읽거나, 마이크로 녹음 (녹음 시간은 옵션으로 계산, 녹음은 wav 로 저장)"""
    if args.wav:
        return read_wav(args.wav, args.fs)
    if not args.record:                                    # 보내는 소리 길이 + 여유 12초
        args.record = float(np.ceil(tx_duration(model, args)) + 12)
    sig = record(args.record, args.fs, args.device)
    args.save_wav = args.save_wav or os.path.join(codec.RESULT_DIR, "rx_record.wav")
    os.makedirs(os.path.dirname(os.path.abspath(args.save_wav)), exist_ok=True)
    wavfile.write(args.save_wav, args.fs, (sig * 32767).astype(np.int16))
    print("녹음 저장:", args.save_wav, f" (다시 복조: python rx.py --wav {args.save_wav} + 같은 옵션)")
    return sig


def read_sender_settings(frames, frame_bits):
    """받은 프레임들에서 보내는 쪽 설정 읽기 -> (B, 조각 배치, 사진 지문 또는 None)
    - B, 전체 프레임 수: 모든 프레임 헤더의 다수결
    - 조각 배치, 지문: 첫 프레임 맨 앞의 사진 정보. 프레임 수와 맞을 때만 믿고, 아니면 프레임 수로 정사각형 추정"""
    B = int(np.bincount([f['B'] for f in frames]).argmax())
    total = int(np.bincount([f['total'] for f in frames]).argmax())
    first = [f for f in frames if f['seq'] == 0 and len(f['payload']) >= codec.INFO_LEN]
    if first:
        img_id, (gh, gw) = codec.parse_info(first[0]['payload'])
        need = int(np.ceil((codec.INFO_LEN + 512 * B * gh * gw) / frame_bits))
        if 1 <= gh <= 15 and 1 <= gw <= 15 and need == total:
            return B, (gh, gw), img_id
    t = max(1, int(round(((total * frame_bits - codec.INFO_LEN) / (512 * B)) ** 0.5)))
    return B, (t, t), None


def match_sender(sig, args, model):
    """보내는 쪽 설정(B, 조각 배치)에 맞춘다. args.grid, args.sent_id 를 정하고 (B 가 다르면 다시 부른) 모델을 돌려준다"""
    frames = modem.decode_frames(sig, args)
    if not frames:                                         # 아무것도 못 받음: 받는 쪽 설정 그대로
        args.grid, args.sent_id = (args.tiles, args.tiles), None
        return model
    B, args.grid, args.sent_id = read_sender_settings(frames, args.frame_bits)
    gh, gw = args.grid
    print(f"[자동] 사진 크기 {32 * gw}x{32 * gh} (32x32 조각 가로 {gw} x 세로 {gh})")
    if B in (2, 4, 8) and B != args.B:
        print(f"[자동] 보내는 쪽은 B={B} 입니다 (받는 쪽 설정 B={args.B}). 맞춰서 복원합니다.")
        args.B, args.ckpt = B, None
        model, _ = codec.load_model(args)
    return model


def load_original(args):
    """비교할 원본: --image / --index 로 주거나, 없으면 보낸 사진의 지문으로 images 폴더에서 찾는다. 없으면 None"""
    if not args.image and args.index is None and args.sent_id is not None:
        args.image = codec.find_image(args.sent_id)
        print(f"[자동] 보낸 사진: {os.path.basename(args.image)} (images 폴더에서 찾음) -> 원본과 비교합니다"
              if args.image else "[자동] 보낸 사진이 이 컴퓨터의 images 폴더에 없음 -> 받은 사진만 보여 줍니다")
    if args.image or args.index is not None:
        return codec.load_image(args)                      # args.grid (보내는 쪽 배치) 로 자름
    return None


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
    sig = get_signal(args, model)                          # 1. 소리 받기
    model = match_sender(sig, args, model)                 # 2. 보내는 쪽 설정에 맞추기
    x_true = load_original(args)                           # 3. 비교할 원본 (없으면 None)
    rep, x_rx, x_clean = receive(sig, model, x_true, args) # 4. 복원
    print_report(rep)
    save_figure(x_true, x_clean, x_rx, rep, args.fig, show=not args.no_show)


if __name__ == "__main__":
    main()
