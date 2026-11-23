"""
이미지 <-> 비트열 (Week 1 모델 재사용)

송신:  이미지 -> Encoder -> z -> 양자화 인덱스 -> 비트열
수신:  비트열 -> 인덱스 -> z_q -> Decoder -> 이미지
       (못 받은 프레임의 latent는 0으로 채움: tanh 출력의 가운데 값)
"""
import os
import sys
import numpy as np
import torch

# Week 1 코드(모델 정의, 체크포인트)를 그대로 가져다 쓴다
HERE = os.path.dirname(os.path.abspath(__file__))
WEEK1 = os.path.normpath(os.path.join(HERE, "..", "week1"))
RESULT_DIR = os.path.join(HERE, "results")
sys.path.insert(0, WEEK1)
import utils                                              # noqa: E402
from models.channel import index_to_bits, bits_to_index   # noqa: E402
from models.metrics import PSNR                           # noqa: E402


def add_codec_args(parser):
    g = parser.add_argument_group("image codec")
    g.add_argument("--ckpt", type=str, default=os.path.join(utils.CKPT_DIR, "uniform_B4.pth"),
                   help="Week 1 체크포인트 (기본: week1/checkpoints/uniform_B4.pth)")
    g.add_argument("--index", type=int, default=0, help="CIFAR-100 테스트 이미지 번호")
    g.add_argument("--image", type=str, default=None, help="직접 고른 이미지 파일 (32x32로 축소)")
    g.add_argument("--gray", action="store_true", help="Gray 코드 비트 매핑")
    g.add_argument("--interleave", action="store_true",
                   help="latent 순서를 섞어 보냄 (프레임 손실이 이미지 한 곳에 몰리지 않게)")
    return parser


def interleaver(n, enable):
    # 송수신이 같은 시드로 같은 순열을 만든다. enable=False면 항등 순열
    return np.random.default_rng(1234).permutation(n) if enable else np.arange(n)


def load_image(args):
    """(1,3,32,32) 텐서, 값 범위 [-0.5, 0.5]"""
    if args.image:
        from PIL import Image
        img = Image.open(args.image).convert("RGB").resize((32, 32))
        return utils.TRANSFORM(img).unsqueeze(0)
    import torchvision.datasets as datasets
    utils.ensure_cifar100()                     # 없으면 download_data.py 로 받기 (이어받기 지원)
    val = datasets.CIFAR100(root=utils.DATA_ROOT, train=False, download=False, transform=utils.TRANSFORM)
    return val[args.index][0].unsqueeze(0)


@torch.no_grad()
def image_to_bits(model, x, gray=False, interleave=False):
    """이미지 -> 비트열(numpy uint8), latent 모양"""
    sq = model.scalar_quantization
    z = model.encode(x).permute(0, 2, 3, 1)          # (1, H, W, C)
    idx = sq.quantize_index(z)
    perm = interleaver(idx.numel(), interleave)
    return index_to_bits(idx.flatten()[perm], sq.B, gray).numpy(), tuple(idx.shape)


@torch.no_grad()
def bits_to_image(model, bits, shape, valid=None, gray=False, interleave=False):
    """비트열 -> 복원 이미지. valid(비트별 수신 여부)가 False인 latent는 0으로"""
    sq = model.scalar_quantization
    n = int(np.prod(shape))
    perm = interleaver(n, interleave)
    idx = torch.zeros(n, dtype=torch.long)
    idx[perm] = bits_to_index(torch.from_numpy(bits.astype(np.int64)), sq.B, (n,), gray)
    z_q = sq.dequantize(idx.view(shape))
    if valid is not None:
        ok = np.zeros(n, dtype=bool)
        ok[perm] = valid.reshape(-1, sq.B).all(axis=1)   # 인터리빙 되돌리기
        z_q = torch.where(torch.from_numpy(ok).view(shape), z_q, torch.zeros_like(z_q))
    return model.decoder(z_q.permute(0, 3, 1, 2).contiguous())


def psnr(x_hat, x):
    return PSNR()(x_hat, x, min_val=-0.5, max_val=0.5)


def to_img(x):
    return (x[0].clamp(-0.5, 0.5) + 0.5).permute(1, 2, 0).numpy()
