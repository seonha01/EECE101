"""
양자화 인덱스 <-> 비트스트림 변환과 BSC(Binary Symmetric Channel) 모델.

흐름:  z_q 인덱스 (정수, 0 ~ 2^B-1)
        -> index_to_bits   : 각 인덱스를 B비트로 펼침 (MSB 먼저)
        -> bsc             : 각 비트를 확률 p로 뒤집음
        -> bits_to_index   : 다시 B비트씩 묶어 정수 인덱스로

★ TODO 채우는 법: `raise NotImplementedError(...)` 줄을 **지우고** 그 자리에 코드를 쓴다.
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
    shifts = torch.arange(B - 1, -1, -1, device=idx.device)   # [B-1, ..., 1, 0]  (MSB 먼저)
    # TODO 5-1: 각 인덱스의 비트를 꺼내 (N, B) 모양의 0/1 텐서 bits 만들기
    #   힌트) 정수 k의 i번째 비트 = (k >> i) & 1
    #         idx.unsqueeze(1) 은 (N, 1), shifts 는 (B,) -> 브로드캐스팅으로 (N, B)
    #   예) idx=5, B=4 -> [0, 1, 0, 1]
    #   ▶ 아래 raise 줄을 지우고 그 자리에 코드를 쓰기.
    #     결과는 반드시 `bits` 라는 이름의 변수에 담기 (맨 아래 return 줄이 bits 를 씀)
    raise NotImplementedError("TODO 5-1 index_to_bits")
    return bits.flatten().to(torch.uint8)


def bits_to_index(bits, B, shape, gray=False):
    """1차원 비트 텐서 -> 정수 인덱스 텐서 (shape 복원)"""
    bits = bits.long().view(-1, B)
    weights = 2 ** torch.arange(B - 1, -1, -1, device=bits.device)   # [2^(B-1), ..., 2, 1]
    # TODO 5-2: 각 행(B비트)을 정수로 -> idx (길이 N)
    #   힌트) [0,1,0,1] -> 0*8 + 1*4 + 0*2 + 1*1 = 5.  bits * weights 를 dim=1 로 더하기
    #   ▶ 아래 raise 줄을 지우고 그 자리에 코드를 쓰기.
    #     결과는 반드시 `idx` 라는 이름의 변수에 담기 (아래 gray / return 줄이 idx 를 씀)
    raise NotImplementedError("TODO 5-2 bits_to_index")
    if gray:
        idx = gray_to_binary(idx, B)
    return idx.view(shape)


def bsc(bits, p):
    """Binary Symmetric Channel: 각 비트를 독립적으로 확률 p로 뒤집는다."""
    if p <= 0:
        return bits
    # TODO 5-3: 비트마다 확률 p로 1이 되는 flip 텐서를 만들고 bits와 XOR(^)
    #   힌트) torch.rand(bits.shape, device=bits.device) < p  -> True/False,  .to(bits.dtype) 로 0/1 변환
    #         (device=bits.device 를 빼면 GPU에서 "tensors on different devices" 에러)
    #   ▶ 아래 raise 줄을 지우고, 뒤집힌 비트를 return
    raise NotImplementedError("TODO 5-3 bsc")


def transmit_index(idx, B, p, gray=False):
    """인덱스 -> 비트 -> BSC(p) -> 인덱스 (채널을 통과한 인덱스 반환)"""
    bits = index_to_bits(idx, B, gray)
    bits = bsc(bits, p)
    return bits_to_index(bits, B, idx.shape, gray)
