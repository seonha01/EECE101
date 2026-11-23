"""
송신기 (노트북 A): 이미지 -> Encoder+양자화 -> 비트열 -> BFSK 오디오 -> 스피커 / wav

예)  python tx.py --out tx.wav            # wav로 저장만 (휴대폰으로 옮겨서 재생해도 된다)
     python tx.py --play                   # 스피커로 송신
     python tx.py --B 2 --play             # images 폴더의 최근 사진을 1/24 로 압축해서 송신
     (images 폴더가 비어 있으면 CIFAR-100 테스트 0번 이미지)
"""
import os
import time
import argparse
import numpy as np
from scipy.io import wavfile

import modem
import image_codec as codec
import utils


def make_tx_signal(model, x, cfg):
    """이미지 한 장 -> (오디오 신호, 송신 비트열, 프레임 수)"""
    bits, _ = codec.image_to_bits(model, x, cfg.gray, cfg.interleave)
    sig, n_frames = modem.build_tx_signal(bits, model.scalar_quantization.B, cfg)
    return sig, bits, n_frames


def rx_options(args):
    # 받는 쪽이 같게 맞춰야 하는 옵션만 (기본값과 다른 것)
    opts = [f"--ckpt ../week1/checkpoints/{args.method}_B{args.B}.pth"]
    for key, default in [("baud", 200), ("f0", 3000), ("f1", 5000), ("frame_bits", 256)]:
        if getattr(args, key) != default:
            opts.append(f"--{key} {getattr(args, key):g}")
    opts += [f"--{k}" for k in ("interleave", "gray") if getattr(args, k)]
    return " ".join(opts)


def play(sig, fs, device=None):
    import sounddevice as sd
    sd.play(sig.astype(np.float32), fs, device=device)
    sd.wait()


def main():
    parser = argparse.ArgumentParser()
    modem.add_modem_args(parser)
    codec.add_codec_args(parser)
    parser.add_argument("--out", type=str, default=None, help="wav 파일로 저장")
    parser.add_argument("--play", action="store_true", help="스피커로 재생")
    parser.add_argument("--device", default=None, help="sounddevice 출력 장치 번호/이름")
    parser.add_argument("--wait", type=float, default=3, help="재생 전 카운트다운 (s)")
    args = parser.parse_args()

    if args.image is None and args.index is None and codec.list_images(codec.IMAGE_DIR):
        args.image = codec.IMAGE_DIR                     # 기본: images 폴더의 가장 최근 사진
    model, cfg = codec.load_model(args)
    x = codec.load_image(args)
    sig, bits, n_frames = make_tx_signal(model, x, args)

    dur = len(sig) / args.fs
    raw = x.numel() * 8                                  # 원본: 32x32x3 화소 x 8비트
    print(f"이미지: {args.image or f'CIFAR-100 테스트 {args.index}번'}")
    print(f"압축: 원본 {raw:,} bit -> {len(bits):,} bit  (1/{raw / len(bits):g},  {cfg['method']} B={cfg['B']})")
    print(f"소리: 프레임 {n_frames}개 x {args.frame_bits} bit  |  길이 {dur:.1f} s")
    print("받는 쪽 명령:")
    print("  python rx.py " + rx_options(args) + f" --record {int(np.ceil(dur)) + 8}")

    if args.out:
        wavfile.write(args.out, args.fs, (sig * 32767).astype(np.int16))
        print("저장:", args.out)
    if args.play:
        print("받는 쪽 녹음을 먼저 시작해 두세요")
        for i in range(int(args.wait), 0, -1):
            print(f"  {i}...", flush=True)
            time.sleep(1)
        print("재생 중...")
        play(sig, args.fs, args.device)
        print("재생 완료")
    if not (args.out or args.play):
        print("--out 또는 --play 를 지정하세요")


if __name__ == "__main__":
    main()
