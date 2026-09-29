"""
보고서 3번용: 양자화 레벨이 latent z 분포의 어디에 놓이는지 그림으로 보기

    python show_levels.py            # B=4
    python show_levels.py --B 2

- 회색 막대: 학습한 모델의 latent z 분포 (checkpoints/uniform_B{B}.pth 가 있을 때)
- 세로선: 방법별 복원 레벨 (내가 채운 함수로 계산)
  uniform / mulaw 는 공식만으로, lloydmax / learned 는 학습된 체크포인트가 있어야 그려짐
그림: results/levels_B{B}.png
"""
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils

CFG = dict(n_hiddens=128, n_residual_hiddens=32, n_residual_layers=2,
           n_embeddings=256, embedding_dim=8, beta=0.25, m=2)


def get_quantizer(method, B):
    # 체크포인트가 있으면 학습된 레벨을, 없으면 (uniform/mulaw는 공식이라 상관없음) 새로 만든 것
    path = os.path.join(utils.CKPT_DIR, f"{method}_B{B}.pth")
    if os.path.exists(path):
        return utils.load_checkpoint(path)[0].scalar_quantization, True
    return utils.build_model(dict(CFG, method=method, B=B)).scalar_quantization, False


@torch.no_grad()
def latent_samples(B, n_batches=10):
    path = os.path.join(utils.CKPT_DIR, f"uniform_B{B}.pth")
    if not os.path.exists(path):
        return None
    model, _ = utils.load_checkpoint(path)
    _, _, _, val_loader, _ = utils.load_data_and_data_loaders('CIFAR100', 256)
    zs = [model.encode(x).flatten() for i, (x, _) in zip(range(n_batches), val_loader)]
    return torch.cat(zs).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    args = parser.parse_args()
    B = args.B

    z = latent_samples(B)
    methods = ['uniform', 'mulaw', 'lloydmax', 'learned']
    fig, axes = plt.subplots(len(methods), 1, figsize=(9, 8), sharex=True)
    for ax, method in zip(axes, methods):
        sq, trained = get_quantizer(method, B)
        if method in ('lloydmax', 'learned') and not trained:
            ax.set_title(f"{method}: checkpoints/{method}_B{B}.pth 없음 (먼저 학습)")
            continue
        levels = sq.dequantize(torch.arange(2 ** B)).detach().numpy()
        if z is not None:
            ax.hist(z, bins=200, range=(-1, 1), color="lightgray", density=True)
        for v in levels:
            ax.axvline(v, color=f"C{methods.index(method)}", lw=1)
        ax.set_title(f"{method}  (B={B}, {2 ** B} levels)")
        ax.set_yticks([])
        print(f"{method:9s} 레벨:", np.round(levels, 3))
    axes[-1].set_xlabel("z value (tanh output, -1 ~ 1)")
    if z is None:
        print(f"(checkpoints/uniform_B{B}.pth 가 없어서 z 분포는 생략)")
    os.makedirs(utils.RESULT_DIR, exist_ok=True)
    path = os.path.join(utils.RESULT_DIR, f"levels_B{B}.png")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print("그림 저장:", path)


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print(f"아직 안 채운 함수가 있어요: {e}  -> check_week1.py 를 먼저 통과하세요")
