"""
송신기 (노트북 A): 이미지 -> Encoder+양자화 -> 비트열 -> BFSK 오디오 -> 스피커 / wav

예)  python tx.py --out tx.wav            # wav로 저장만 (휴대폰으로 옮겨서 재생해도 된다)
     python tx.py --play                   # 스피커로 송신
"""
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
    args = parser.parse_args()

    model, cfg = utils.load_checkpoint(args.ckpt)
    x = codec.load_image(args)
    sig, bits, n_frames = make_tx_signal(model, x, args)

    dur = len(sig) / args.fs
    print(f"모델: {cfg['method']}, B={cfg['B']}  |  비트 {len(bits)}  |  프레임 {n_frames}개 "
          f"x {args.frame_bits} bit  |  길이 {dur:.1f} s")
    print(f"받는 쪽 녹음 시간은 넉넉히:  --record {int(np.ceil(dur)) + 8}   (송신 길이 + 시작 버튼 누르는 시간)")

    if args.out:
        wavfile.write(args.out, args.fs, (sig * 32767).astype(np.int16))
        print("저장:", args.out)
    if args.play:
        print("재생 중... (수신기 녹음을 먼저 시작해 두세요)")
        play(sig, args.fs, args.device)
        print("재생 완료")
    if not (args.out or args.play):
        print("--out 또는 --play 를 지정하세요")


if __name__ == "__main__":
    main()
