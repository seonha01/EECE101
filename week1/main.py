"""
Week 1: Autoencoder + 스칼라 양자화 학습 / 테스트

예)  python main.py --method uniform --B 4 --n_updates 3000
     python main.py --method lloydmax --B 4 --test_only      # checkpoints/lloydmax_B4.pth 사용
"""
import os
import numpy as np
import torch
import torch.optim as optim
import argparse
import utils
from models.metrics import PSNR

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_updates", type=int, default=3000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=8)    # latent 채널 수 C
parser.add_argument("--n_embeddings", type=int, default=256)
parser.add_argument("--m", type=int, default=2)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--dataset", type=str, default='CIFAR100')
# ---- 양자화 설정 ----
parser.add_argument("--method", type=str, default='uniform',
                    choices=['uniform', 'mulaw', 'lloydmax', 'learned'])
parser.add_argument("--B", type=int, default=4)
parser.add_argument("--ckpt", type=str, default=None, help="저장/불러올 체크포인트 경로")
parser.add_argument("--test_only", action="store_true", help="학습 없이 ckpt로 테스트만")
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto = GPU 있으면 GPU, 없으면 CPU")
parser.add_argument("--threads", type=int, default=0, help="CPU 스레드 수 (0=기본)")


def model_device(model):
    # 모델이 올라가 있는 장치 (데이터도 같은 장치로 보내야 함)
    return next(model.parameters()).device


def train(model, optimizer, loader, args):
    model.train()
    device = model_device(model)
    it = utils.infinite_loader(loader)
    log = {'MSE': [], 'PSNR': [], 'perp': []}
    for i in range(args.n_updates):
        x, _ = next(it)
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity, _, _ = model(x)
        recon_loss = torch.mean((x_hat - x) ** 2)
        loss = recon_loss + embedding_loss
        loss.backward()
        optimizer.step()

        log['MSE'].append(recon_loss.item())
        log['PSNR'].append(PSNR()(x_hat, x, min_val=-0.5, max_val=0.5))
        log['perp'].append(perplexity.item())
        if (i + 1) % args.log_interval == 0:
            n = args.log_interval
            print(f"Update #{i+1}  MSE {np.mean(log['MSE'][-n:]):.5f}  "
                  f"PSNR {np.mean(log['PSNR'][-n:]):.2f} dB  "
                  f"Perplexity {np.mean(log['perp'][-n:]):.2f}")


@torch.no_grad()
def calculate_z_values(model, loader, max_batches=200):
    # 학습 데이터의 latent z (tanh 출력)를 모은다 -> Lloyd-Max 레벨 추정용
    model.eval()
    device = model_device(model)
    zs = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        zs.append(model.encode(x.to(device)).flatten())
    return torch.cat(zs)


@torch.no_grad()
def test(model, loader, p=0.0):
    """테스트셋 전체의 평균 MSE, PSNR (채널 BSC(p) 적용 가능)"""
    model.eval()
    device = model_device(model)
    model.scalar_quantization.channel_p = p
    mse, psnr, bits = [], [], 0
    for x, _ in loader:
        x = x.to(device)
        _, x_hat, _, _, bits = model(x)
        mse.append(torch.mean((x_hat.clamp(-0.5, 0.5) - x) ** 2).item())
        psnr.append(PSNR()(x_hat, x, min_val=-0.5, max_val=0.5))
    model.scalar_quantization.channel_p = 0.0
    return float(np.mean(mse)), float(np.mean(psnr)), bits


def main(args):
    device = utils.get_device(args.device)
    print("device:", device)
    _, _, train_loader, val_loader, _ = utils.load_data_and_data_loaders(
        args.dataset, args.batch_size)
    ckpt = args.ckpt or os.path.join(utils.CKPT_DIR, f"{args.method}_B{args.B}.pth")

    if args.test_only:
        model, _ = utils.load_checkpoint(ckpt, device)
    else:
        model = utils.build_model(vars(args)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
        train(model, optimizer, train_loader, args)
        print("training over")
        if args.method == 'lloydmax':
            # 학습 후 전체 latent 분포로 Lloyd-Max 레벨을 최종 추정
            levels = model.scalar_quantization.fit_lloyd_max(calculate_z_values(model, train_loader))
            print("Lloyd-Max levels:", np.round(levels.cpu().numpy(), 3))
        utils.save_checkpoint(model, args, ckpt)
        print("saved:", ckpt)

    mse, psnr, bits = test(model, val_loader)
    raw_bits = 3 * 32 * 32 * 8
    print(f"Test  MSE {mse:.5f}  PSNR {psnr:.2f} dB  | 전송 비트/이미지 {bits} "
          f"(원본 {raw_bits} bit, 압축률 {raw_bits / bits:.1f}x)")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    main(args)
