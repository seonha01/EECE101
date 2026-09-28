import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.channel import transmit_index


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def uniform_levels(B):
    # [-1, 1]을 2^B 칸으로 나눈 각 칸의 가운데 값 (mid-rise 균일 양자화 레벨)
    L = 2 ** B
    step = 2.0 / L
    return torch.linspace(-1 + step / 2, 1 - step / 2, L)


def midpoints(levels):
    # 인접 레벨의 중간점 = 결정 경계 (nearest-neighbor 규칙)
    return (levels[:-1] + levels[1:]) / 2


class ScalarQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    - method : 'uniform' | 'mulaw' | 'lloydmax' | 'learned'
    - B : 양자화 비트 수 (레벨 수 = 2^B)
    - mu : mu-law 파라미터
    """

    METHODS = ('uniform', 'mulaw', 'lloydmax', 'learned')

    def __init__(self, n_e, e_dim, beta, method='uniform', B=4, mu=255.0):
        super(ScalarQuantizer, self).__init__()
        assert method in self.METHODS, f"method는 {self.METHODS} 중 하나"
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.method = method
        self.B = B
        self.mu = mu

        # Lloyd-Max 레벨: 역전파로 학습하지 않고 latent 분포로부터 추정 -> buffer
        self.register_buffer('lm_levels', uniform_levels(B))
        # 직접 설계한 양자화기: 레벨 자체를 학습 가능한 파라미터로 둔다
        if method == 'learned':
            self.levels = nn.Parameter(uniform_levels(B))

        # 채널 설정 (평가 때 바깥에서 바꿔 쓴다)
        self.channel_p = 0.0    # BSC 비트 반전 확률
        self.gray = False       # Gray 코드 매핑 사용 여부

    ##############################WCML PROJECT (START)######################################
    # 모든 양자화기를 "z -> 인덱스(송신 심볼) -> 복원값" 두 단계로 나눠 구현한다.
    # 인덱스 단계가 있어야 비트스트림으로 바꿔 채널에 보낼 수 있기 때문.

    # ---------- 1) Uniform ----------
    def uniform_index(self, z, B):
        # z in [-1, 1] -> 0 ~ 2^B-1 칸 번호
        # TODO 1-1: [-1, 1]을 L = 2^B 칸으로 똑같이 나눴을 때 z가 몇 번째 칸인지 구하기
        #   힌트) (z + 1) / 2 는 [0, 1] 범위 -> 여기에 L을 곱하고 torch.floor
        #         z = 1 이면 L이 나오므로 .clamp(0, L - 1) 로 잘라주고, 마지막에 .long()
        #   예) B=2: z=-0.6 -> 0,  z=0.2 -> 2,  z=1.0 -> 3
        raise NotImplementedError("TODO 1-1 uniform_index")

    def uniform_value(self, idx, B):
        # 칸 번호 -> 칸의 가운데 값
        # TODO 1-2: 칸 폭 step = 2 / 2^B 일 때, k번째 칸의 가운데 값 = -1 + (k + 0.5) * step
        #   힌트) idx는 정수 텐서이므로 idx.float() 로 바꿔서 계산
        #   예) B=2: [0, 1, 2, 3] -> [-0.75, -0.25, 0.25, 0.75]
        raise NotImplementedError("TODO 1-2 uniform_value")

    def uniform_quantization(self, z, B):
        z_q = self.uniform_value(self.uniform_index(z, B), B)
        return z_q

    # ---------- 2) mu-law ----------
    def mu_compress(self, z):
        # 0 근처를 넓게 펴 주는 압축 함수 F(z) = sgn(z) ln(1+mu|z|) / ln(1+mu)
        # TODO 2-1: 위 식을 한 줄로
        #   힌트) torch.sign, torch.log1p(x) = ln(1+x), z.abs(), np.log1p(self.mu)
        raise NotImplementedError("TODO 2-1 mu_compress")

    def mu_expand(self, y):
        # 압축의 역함수 F^-1(y) = sgn(y) ((1+mu)^|y| - 1) / mu
        # TODO 2-2: 위 식을 한 줄로  (mu_expand(mu_compress(z)) == z 가 되어야 함)
        raise NotImplementedError("TODO 2-2 mu_expand")

    def mu_law_quantization(self, z, B):
        # 압축 -> 균일 양자화 -> 신장
        idx = self.uniform_index(self.mu_compress(z), B)
        z_q = self.mu_expand(self.uniform_value(idx, B))
        return z_q

    # ---------- 3) Lloyd-Max ----------
    @torch.no_grad()
    def lloyd_max_step(self, samples, levels):
        # Lloyd 알고리즘 1회: (a) 경계 = 레벨 중간점  (b) 레벨 = 각 구간 샘플의 평균
        # (a) 각 샘플이 몇 번째 구간(레벨)에 속하는지 -> idx (0 ~ L-1)
        idx = torch.bucketize(samples, midpoints(levels))

        # TODO 3: (b) 구간 k에 속한 샘플들의 평균을 새 레벨 k로
        #   - 구간에 샘플이 하나도 없으면 원래 레벨을 그대로 둔다
        #   힌트) sums = torch.zeros_like(levels).index_add_(0, idx, samples)            # 구간별 합
        #         cnts = torch.zeros_like(levels).index_add_(0, idx, torch.ones_like(samples))  # 구간별 개수
        #         new_levels = torch.where(cnts > 0, 합 / 개수, levels)   (0으로 나누지 않게 cnts.clamp(min=1))
        #   (for k in range(L) 반복문으로 풀어도 된다. 다만 B=8이면 느림)
        #   예) samples=[-1, -0.8, 0.5, 0.7], levels=[-0.5, 0.5] -> [-0.9, 0.6]
        raise NotImplementedError("TODO 3 lloyd_max_step")
        return torch.sort(new_levels).values

    @torch.no_grad()
    def fit_lloyd_max(self, samples, n_iter=100, init_quantile=True):
        """학습 latent 샘플(1차원)로 Lloyd-Max 레벨을 추정해 self.lm_levels에 저장"""
        samples = samples.flatten().float()
        L = self.lm_levels.numel()
        levels = self.lm_levels.clone()
        if init_quantile:  # 분위수로 초기화하면 빈 구간이 거의 안 생긴다
            q = (torch.arange(L, device=samples.device) + 0.5) / L
            levels = torch.quantile(samples[:1_000_000], q.to(samples.dtype))
        for _ in range(n_iter):
            levels = self.lloyd_max_step(samples, levels)
        self.lm_levels.copy_(levels)
        return levels

    def lloyd_index(self, z):
        return torch.bucketize(z.contiguous(), midpoints(self.lm_levels))

    def Loyld_max_quantization(self, z, B):
        assert 2 ** B == self.lm_levels.numel(), "학습된 레벨 수와 B가 다릅니다"
        z_q = self.lm_levels[self.lloyd_index(z)]
        return z_q

    # ---------- 4) Your own: 학습 가능한 레벨 ----------
    # 레벨 L개를 nn.Parameter로 두고, VQ-VAE처럼 codebook/commitment loss로 학습.
    # (Lloyd-Max를 역전파로 end-to-end 학습하는 버전이라고 생각하면 된다)
    def sorted_levels(self):
        return torch.sort(self.levels).values

    def learned_index(self, z):
        return torch.bucketize(z.contiguous(), midpoints(self.sorted_levels().detach()))

    def your_own_quantization(self, z, B):
        assert 2 ** B == self.levels.numel(), "학습된 레벨 수와 B가 다릅니다"
        z_q = self.sorted_levels()[self.learned_index(z)]
        return z_q

    def level_loss(self, z, z_q):
        # learned 방법에서만: 레벨을 z 쪽으로(codebook), z를 레벨 쪽으로(commitment)
        if self.method != 'learned':
            return torch.tensor(0.0, device=z.device)
        # TODO 4: loss = codebook + beta * commitment
        #   codebook   = mean((z_q - sg[z])^2)   -> 레벨만 움직임 (z는 고정)
        #   commitment = mean((sg[z_q] - z)^2)   -> 인코더만 움직임 (레벨은 고정)
        #   sg[x] (stop-gradient) 는 PyTorch에서 x.detach()
        raise NotImplementedError("TODO 4 level_loss")

    # ---------- 공통: 인덱스 <-> 값 ----------
    def quantize_index(self, z):
        """z -> 정수 인덱스 (이 인덱스를 B비트로 바꾼 것이 실제 전송 비트)"""
        if self.method == 'uniform':
            return self.uniform_index(z, self.B)
        if self.method == 'mulaw':
            return self.uniform_index(self.mu_compress(z), self.B)
        if self.method == 'lloydmax':
            return self.lloyd_index(z)
        return self.learned_index(z)

    def dequantize(self, idx):
        """정수 인덱스 -> 복원값 z_q"""
        if self.method == 'uniform':
            return self.uniform_value(idx, self.B)
        if self.method == 'mulaw':
            return self.mu_expand(self.uniform_value(idx, self.B))
        if self.method == 'lloydmax':
            return self.lm_levels[idx]
        return self.sorted_levels()[idx]
    #############################WCML PROJECT (END)#########################################

    def perplexity(self, idx):
        # 인덱스 사용 분포의 perplexity = 2^(엔트로피). 최대값은 2^B (모든 레벨 고르게 사용)
        p = torch.bincount(idx.flatten(), minlength=2 ** self.B).float()
        p = p / p.sum()
        return torch.exp(-(p * torch.log(p + 1e-10)).sum())

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()

        ##############################WCML PROJECT (START)######################################
        # 아래 4줄과 같은 일을 "인덱스 -> (채널) -> 값" 경로로 수행한다.
        # z_q = self.uniform_quantization(z, B=self.B)
        # z_q = self.mu_law_quantization(z, B=self.B)
        # z_q = self.Loyld_max_quantization(z, B=self.B)
        # z_q = self.your_own_quantization(z, B=self.B)
        if self.method == 'lloydmax' and self.training:
            self.lm_levels.copy_(self.lloyd_max_step(z.detach().flatten(), self.lm_levels))

        idx = self.quantize_index(z)                  # 송신할 정수 심볼
        if self.channel_p > 0:                        # 비트스트림 -> BSC(p) -> 인덱스
            idx = transmit_index(idx, self.B, self.channel_p, self.gray)
        z_q = self.dequantize(idx)                    # 수신 측 복원값
        loss = self.level_loss(z, z_q)
        #############################WCML PROJECT (END)#########################################

        # preserve gradients
        # 질문) 이 줄은 무슨 뜻이고, 무슨 역할을 하며, 왜 필요할까?  (README의 "생각해 볼 것" 참고)
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.tensor(0).to(z.device)
        perplexity = self.perplexity(idx)

        # 이미지 1장당 전송 비트 수 = (H * W * C) * B
        bits_per_image = z[0].numel() * self.B

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, e_mean, bits_per_image
