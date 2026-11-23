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
IMAGE_DIR = os.path.join(HERE, "images")          # tx.py 가 기본으로 보내는 사진 폴더
sys.path.insert(0, WEEK1)
import utils                                              # noqa: E402
from models.channel import index_to_bits, bits_to_index   # noqa: E402
from models.metrics import PSNR                           # noqa: E402


def add_codec_args(parser):
    g = parser.add_argument_group("image codec")
    g.add_argument("--B", type=int, default=4, choices=[2, 4, 8],
                   help="latent 값 하나당 비트 수 = 압축률 (2: 1/24, 4: 1/12, 8: 1/6)")
    g.add_argument("--method", type=str, default="uniform", choices=["uniform", "mulaw", "lloydmax", "learned"],
                   help="1주차 양자화 방법")
    g.add_argument("--ckpt", type=str, default=None,
                   help="체크포인트 직접 지정 (기본: week1/checkpoints/{method}_B{B}.pth)")
    g.add_argument("--index", type=int, default=None, help="CIFAR-100 테스트 이미지 번호 (기본 0)")
    g.add_argument("--image", type=str, default=None,
                   help="직접 고른 사진 파일 (가운데를 정사각형으로 잘라 32x32로 축소). 폴더를 주면 가장 최근 사진")
    g.add_argument("--tiles", type=int, default=1,
                   help="해상도: 사진을 (32*tiles)x(32*tiles) 로 줄이고 32x32 조각 tiles x tiles 개로 나눠 보냄 (송수신 같게)")
    g.add_argument("--gray", action="store_true", help="Gray 코드 비트 매핑")
    g.add_argument("--interleave", action="store_true",
                   help="latent 순서를 섞어 보냄 (프레임 손실이 이미지 한 곳에 몰리지 않게)")
    return parser


def interleaver(n, enable):
    # 송수신이 같은 시드로 같은 순열을 만든다. enable=False면 항등 순열
    return np.random.default_rng(1234).permutation(n) if enable else np.arange(n)


def load_model(args):
    """--ckpt 가 없으면 --method, --B 로 체크포인트를 고른다. (model, 설정) 반환"""
    if args.ckpt is None:
        args.ckpt = os.path.join(utils.CKPT_DIR, f"{args.method}_B{args.B}.pth")
    model, cfg = utils.load_checkpoint(args.ckpt)
    args.B, args.method = cfg['B'], cfg['method']        # --ckpt 로 직접 준 경우에도 맞춰 둠
    return model, cfg


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]


def newest_image(folder):
    files = list_images(folder)
    if not files:
        raise SystemExit(f"{folder} 폴더에 사진이 없습니다")
    return max(files, key=os.path.getmtime)


def load_image(args):
    """(1,3,S,S) 텐서 (S = 32 x tiles), 값 범위 [-0.5, 0.5]"""
    size = 32 * getattr(args, "tiles", 1)
    if args.image:
        from PIL import Image
        if not os.path.exists(args.image) and os.path.exists(os.path.join(IMAGE_DIR, args.image)):
            args.image = os.path.join(IMAGE_DIR, args.image)    # "--image Lenna.png" -> images/Lenna.png
        if os.path.isdir(args.image):
            args.image = newest_image(args.image)
        img = Image.open(args.image).convert("RGB")
        w, h = img.size
        s = min(w, h)                                   # 가운데 정사각형만 (찌그러지지 않게)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        img = img.resize((size, size), Image.LANCZOS)
        return utils.TRANSFORM(img).unsqueeze(0)
    if args.index is None:
        args.index = 0
    import torchvision.datasets as datasets
    utils.ensure_cifar100()                     # 없으면 download_data.py 로 받기 (이어받기 지원)
    val = datasets.CIFAR100(root=utils.DATA_ROOT, train=False, download=False, transform=utils.TRANSFORM)
    x = val[args.index][0].unsqueeze(0)
    return x if size == 32 else torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear")


def to_patches(x):
    """(1,3,32N,32N) -> (N*N,3,32,32): 왼쪽 위부터 가로로 조각내기"""
    n = x.shape[-1] // 32
    return x.reshape(1, 3, n, 32, n, 32).permute(0, 2, 4, 1, 3, 5).reshape(n * n, 3, 32, 32)


def from_patches(p):
    """(N*N,3,32,32) -> (1,3,32N,32N): 조각 다시 붙이기"""
    n = int(round(p.shape[0] ** 0.5))
    return p.reshape(n, n, 3, 32, 32).permute(2, 0, 3, 1, 4).reshape(1, 3, 32 * n, 32 * n)


@torch.no_grad()
def image_to_bits(model, x, gray=False, interleave=False):
    """이미지 -> 비트열(numpy uint8), latent 모양. 32x32 보다 크면 32x32 조각마다 인코딩"""
    sq = model.scalar_quantization
    z = model.encode(to_patches(x)).permute(0, 2, 3, 1)   # (조각 수, H, W, C)
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
    return from_patches(model.decoder(z_q.permute(0, 3, 1, 2).contiguous()))


def psnr(x_hat, x):
    return PSNR()(x_hat, x, min_val=-0.5, max_val=0.5)


def to_img(x):
    return (x[0].clamp(-0.5, 0.5) + 0.5).permute(1, 2, 0).numpy()
