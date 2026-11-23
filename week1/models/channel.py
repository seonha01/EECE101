"""
양자화 인덱스 <-> 비트스트림 변환과 BSC(Binary Symmetric Channel) 모델.

흐름:  z_q 인덱스 (정수, 0 ~ 2^B-1)
        -> index_to_bits   : 각 인덱스를 B비트로 펼침 (MSB 먼저)
        -> bsc             : 각 비트를 확률 p로 뒤집음
        -> bits_to_index   : 다시 B비트씩 묶어 정수 인덱스로
"""
import torch


def binary_to_gray(idx):
    # Gray 코드: 인접한 레벨끼리 1비트만 다르게 만든다 -> 비트 에러 시 값이 덜 튄다
    return idx ^ (idx >> 1)


def gray_to_binary(g, B):
    # Gray -> 이진수 복원 (상위 비트부터 XOR 누적)
    idx = g.clone()
    shift = 1
    while shift < B:
        idx = idx ^ (idx >> shift)
        shift *= 2
    return idx


def index_to_bits(idx, B, gray=False):
    """정수 인덱스 텐서 (임의 shape) -> 1차원 비트 텐서 (길이 = idx.numel() * B)"""
    idx = idx.long().flatten()
    if gray:
        idx = binary_to_gray(idx)
    shifts = torch.arange(B - 1, -1, -1, device=idx.device)   # MSB 먼저
    bits = (idx.unsqueeze(1) >> shifts) & 1                    # (N, B)
    return bits.flatten().to(torch.uint8)


def bits_to_index(bits, B, shape, gray=False):
    """1차원 비트 텐서 -> 정수 인덱스 텐서 (shape 복원)"""
    bits = bits.long().view(-1, B)
    weights = 2 ** torch.arange(B - 1, -1, -1, device=bits.device)
    idx = (bits * weights).sum(dim=1)
    if gray:
        idx = gray_to_binary(idx, B)
    return idx.view(shape)


def bsc(bits, p):
    """Binary Symmetric Channel: 각 비트를 독립적으로 확률 p로 뒤집는다."""
    if p <= 0:
        return bits
    flip = (torch.rand(bits.shape, device=bits.device) < p).to(bits.dtype)
    return bits ^ flip


def transmit_index(idx, B, p, gray=False):
    """인덱스 -> 비트 -> BSC(p) -> 인덱스 (채널을 통과한 인덱스 반환)"""
    bits = index_to_bits(idx, B, gray)
    bits = bsc(bits, p)
    return bits_to_index(bits, B, idx.shape, gray)
