import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import numpy as np

# 경로는 이 파일 위치 기준 (어느 폴더에서 실행해도 동일)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(os.path.join(HERE, "..", "data"))   # 저장소 최상위 data/ (공용)
CKPT_DIR = os.path.join(HERE, "checkpoints")
RESULT_DIR = os.path.join(HERE, "results")


def get_device(name="auto"):
    """'auto'면 NVIDIA GPU(cuda)가 있을 때 GPU, 없으면 CPU. 'cpu' / 'cuda'로 직접 고를 수도 있음"""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다. --device cpu 로 실행하세요.")
    return torch.device(name)


# 이미지 정규화: [0,1] -> [-0.5, 0.5]  (ref 코드와 동일)
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
])


def ensure_cifar100():
    # data 폴더에 CIFAR-100이 없으면 저장소 최상위의 download_data.py 로 받는다 (이어받기 지원)
    if not os.path.exists(os.path.join(DATA_ROOT, "cifar-100-python", "test")):
        import sys
        sys.path.insert(0, os.path.dirname(DATA_ROOT))
        import download_data
        download_data.main()


def load_CIFAR100(root=DATA_ROOT):
    ensure_cifar100()
    train = datasets.CIFAR100(root=root, train=True, download=False, transform=TRANSFORM)
    val = datasets.CIFAR100(root=root, train=False, download=False, transform=TRANSFORM)
    return train, val


def data_loaders(train_data, val_data, batch_size):
    pin = torch.cuda.is_available()      # GPU로 보낼 때 조금 빨라짐 (CPU만 쓰면 의미 없음)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=pin)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size):
    if dataset != 'CIFAR100':
        raise ValueError('Invalid dataset: only CIFAR100 is supported.')
    training_data, validation_data = load_CIFAR100()
    training_loader, validation_loader = data_loaders(training_data, validation_data, batch_size)
    x_train_var = np.var(training_data.data / 255.0)
    return training_data, validation_data, training_loader, validation_loader, x_train_var


def infinite_loader(loader):
    # DataLoader를 끝없이 반복 (매 스텝 iter()를 새로 만드는 비용을 없앰)
    while True:
        for batch in loader:
            yield batch


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


# ---------------- 모델 저장 / 불러오기 ----------------
MODEL_KEYS = ['n_hiddens', 'n_residual_hiddens', 'n_residual_layers',
              'n_embeddings', 'embedding_dim', 'beta', 'm', 'method', 'B']


def save_checkpoint(model, args, path):
    # 모델 구조를 다시 만들 수 있도록 하이퍼파라미터를 함께 저장
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cfg = {k: getattr(args, k) for k in MODEL_KEYS}
    torch.save({'model': model.state_dict(), 'config': cfg}, path)


def build_model(cfg):
    from models.vqvae import VQVAE
    return VQVAE(cfg['n_hiddens'], cfg['n_residual_hiddens'], cfg['n_residual_layers'],
                 cfg['n_embeddings'], cfg['embedding_dim'], cfg['beta'], cfg['m'],
                 method=cfg['method'], B=cfg['B'])


def load_checkpoint(path, device='cpu'):
    # map_location: GPU에서 학습한 모델도 CPU에서 (또는 그 반대로) 불러올 수 있게
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt['config']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt['config']
