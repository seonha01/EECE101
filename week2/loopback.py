"""
개발 1단계: numpy 채널 시뮬레이션 loopback (스피커/마이크 없이)

tx 신호 -> simulate_channel(감쇠, 지연, 에코, 클럭 오차, 잡음) -> rx 복조 -> BER / PSNR
SNR 마다 결과 그림을 results/loopback_snr{SNR}.png 로 저장한다.

예)  python loopback.py                                  # SNR 10, 0, -5, -8, -10 dB
     python loopback.py --snr -8 -9 -10 -11 -12           # 에러가 생기기 시작하는 구간 자세히
     python loopback.py --interleave                      # 인터리빙 (loopback 은 한 번만 붙이면 됨)
     python loopback.py --snr 10 --ppm 500 2000           # 두 컴퓨터 시계 차이(ppm)를 키우면?
     python loopback.py --baud 400 --snr 0 -5 -8          # 더 빠르게 보내면?
     python loopback.py -h                                # 모든 옵션 보기
"""
import os
import argparse

import modem
import image_codec as codec
import utils
from tx import make_tx_signal
from rx import receive, print_report, save_figure


def main():
    parser = argparse.ArgumentParser()
    modem.add_modem_args(parser)
    codec.add_codec_args(parser)
    parser.add_argument("--snr", nargs="+", type=float, default=[10, 0, -5, -8, -10], help="잡음 크기 (dB), 여러 개 가능")
    parser.add_argument("--ppm", nargs="+", type=float, default=[50],
                        help="두 컴퓨터의 시계 차이 (ppm = 백만분의 1). 여러 개 가능")
    parser.add_argument("--tag", type=str, default="", help="그림 파일 이름에 붙일 말 (예: --tag baud400)")
    args = parser.parse_args()

    model, _ = codec.load_model(args)
    x = codec.load_image(args)
    sig, bits, n_frames = make_tx_signal(model, x, args)
    print(f"송신: {len(bits)} bit, {n_frames} 프레임, {len(sig) / args.fs:.1f} s"
          + ("  (interleave)" if args.interleave else ""))

    for ppm in args.ppm:
        for snr in args.snr:
            rx_sig = modem.simulate_channel(sig, args.fs, snr_db=snr, ppm=ppm)
            rep, x_rx, x_clean = receive(rx_sig, model, x, args, quiet=True)
            print(f"\n--- SNR {snr:g} dB, 시계 차이 {ppm:g} ppm ---")
            print_report(rep)
            save_figure(x, x_clean, x_rx, rep, os.path.join(codec.RESULT_DIR, fig_name(args, snr, ppm)))


def fig_name(args, snr, ppm):
    # 기본값과 다른 옵션은 파일 이름에 붙여서, 다른 실험 그림을 덮어쓰지 않게 한다
    parts = [f"loopback_snr{snr:g}"]
    if ppm != 50:
        parts.append(f"ppm{ppm:g}")
    for key, default in [("baud", 200), ("frame_bits", 256), ("sync_thr", 0.25), ("f0", 3000), ("f1", 5000)]:
        if getattr(args, key) != default:
            parts.append(f"{key}{getattr(args, key):g}")
    if args.interleave:
        parts.append("inter")
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts) + ".png"


if __name__ == "__main__":
    main()
