"""
Week 1 실험: 양자화 방법 x B x BSC(p) 에 대해 MSE/PSNR 표 + 복원 이미지 그림 저장

먼저 main.py로 checkpoints/{method}_B{B}.pth 를 만들어 둔 뒤 실행.
예)  python week1_experiment.py
     python week1_experiment.py --methods uniform lloydmax --Bs 4 --gray
"""
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils
from main import test

parser = argparse.ArgumentParser()
parser.add_argument("--methods", nargs="+", default=['uniform', 'mulaw', 'lloydmax', 'learned'])
parser.add_argument("--Bs", nargs="+", type=int, default=[2, 4, 8])
parser.add_argument("--ps", nargs="+", type=float, default=[0.0, 0.01, 0.1])
parser.add_argument("--gray", action="store_true", help="비트 매핑에 Gray 코드 사용")
parser.add_argument("--n_show", type=int, default=8, help="그림에 보일 이미지 수")
parser.add_argument("--out", type=str, default=utils.RESULT_DIR)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto = GPU 있으면 GPU, 없으면 CPU")


def to_img(x):
    # [-0.5, 0.5] 텐서 (3,H,W) -> [0,1] numpy (H,W,3)
    return (x.clamp(-0.5, 0.5) + 0.5).permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def reconstruct(model, x, p, gray):
    model.scalar_quantization.channel_p = p
    model.scalar_quantization.gray = gray
    x_hat = model(x)[1]
    model.scalar_quantization.channel_p = 0.0
    return x_hat


def save_figure(model_by_B, x, ps, gray, path, title):
    # 행: 원본 + (B, p) 조합,  열: 테스트 이미지
    rows = [("Original", x)]
    for B, model in model_by_B.items():
        for p in ps:
            rows.append((f"B={B}\np={p}", reconstruct(model, x, p, gray)))
    n = x.shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.1, len(rows) * 1.15))
    for r, (label, imgs) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(to_img(imgs[c]))
            ax.set_xticks([]); ax.set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_psnr_plot(rows, ps, path):
    # p마다 한 칸, 칸 안에서 방법별 PSNR vs B (방법마다 색 고정)
    fig, axes = plt.subplots(1, len(ps), figsize=(4 * len(ps), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    methods = list(dict.fromkeys(r['method'] for r in rows))
    for ax, p in zip(axes, ps):
        for k, method in enumerate(methods):
            pts = [(r['B'], r['PSNR']) for r in rows if r['method'] == method and r['p'] == p]
            if pts:
                Bs, vals = zip(*pts)
                ax.plot(Bs, vals, marker='o', color=f"C{k}", label=method)
        ax.set_title(f"BSC p = {p}"); ax.set_xlabel("B (bits per latent)")
        ax.set_xticks(sorted({r['B'] for r in rows})); ax.grid(alpha=0.3)
    axes[0].set_ylabel("PSNR (dB)"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def write_table(rows, path):
    lines = ["| method | B | bits/image | p | MSE | PSNR (dB) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['B']} | {r['bits']} | {r['p']} | "
                     f"{r['MSE']:.5f} | {r['PSNR']:.2f} |")
    text = "\n".join(lines)
    open(path, "w", encoding="utf-8").write(text + "\n")
    print(text)


def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = utils.get_device(args.device)
    print("device:", device)
    _, val_data, _, val_loader, _ = utils.load_data_and_data_loaders('CIFAR100', 256)
    x_show = torch.stack([val_data[i][0] for i in range(args.n_show)]).to(device)

    suffix = "_gray" if args.gray else ""
    rows = []
    for method in args.methods:
        model_by_B = {}
        for B in args.Bs:
            path = os.path.join(utils.CKPT_DIR, f"{method}_B{B}.pth")
            if not os.path.exists(path):
                print(f"[skip] {method} B={B}: 학습한 모델 없음 (건너뜀)")
                continue
            model, _ = utils.load_checkpoint(path, device)
            model.scalar_quantization.gray = args.gray
            model_by_B[B] = model
            for p in args.ps:
                torch.manual_seed(args.seed)
                mse, psnr, bits = test(model, val_loader, p)
                rows.append(dict(method=method, B=B, p=p, MSE=mse, PSNR=psnr, bits=bits))
                print(f"{method:9s} B={B} p={p:<5} MSE={mse:.5f} PSNR={psnr:.2f}")
        if model_by_B:
            torch.manual_seed(args.seed)
            save_figure(model_by_B, x_show, args.ps, args.gray,
                        f"{args.out}/recon_{method}{suffix}.png",
                        f"{method} quantizer" + (" (Gray code)" if args.gray else ""))

    write_table(rows, f"{args.out}/table{suffix}.md")
    save_psnr_plot(rows, args.ps, f"{args.out}/psnr{suffix}.png")
    print("saved to", args.out)


if __name__ == "__main__":
    main(parser.parse_args())
