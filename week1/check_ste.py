"""
보고서 2번용 실험: z_q = z + (z_q - z).detach()  (Straight-Through Estimator) 가 없으면 무슨 일이?

    python check_ste.py

같은 이미지 한 묶음으로 loss를 계산하고 역전파(backward)해서,
각 층의 기울기(gradient) 크기를 STE 있을 때 / 없을 때 비교한다.
  - 기울기가 0이면 그 층은 학습이 전혀 안 된다 (optimizer가 움직일 방향이 없음)
"""
import torch
import utils

CFG = dict(n_hiddens=128, n_residual_hiddens=32, n_residual_layers=2,
           n_embeddings=256, embedding_dim=8, beta=0.25, m=2, method='uniform', B=4)


def grad_norms(model, x, use_ste):
    model.zero_grad()
    sq = model.scalar_quantization
    z = model.encode(x).permute(0, 2, 3, 1)             # Encoder 출력 (B, H, W, C)
    z_q = sq.dequantize(sq.quantize_index(z))           # 양자화 (인덱스 -> 값)
    if use_ste:
        z_q = z + (z_q - z).detach()                    # <- 이 한 줄이 있을 때 / 없을 때
    x_hat = model.decoder(z_q.permute(0, 3, 1, 2).contiguous())
    loss = torch.mean((x_hat - x) ** 2)
    loss.backward()

    def norm(module):
        g = [p.grad.norm() ** 2 for p in module.parameters() if p.grad is not None]
        return float(torch.stack(g).sum().sqrt()) if g else 0.0
    return norm(model.encoder), norm(model.decoder)


def main():
    torch.manual_seed(0)
    model = utils.build_model(CFG)
    x = torch.rand(16, 3, 32, 32) - 0.5                 # 이미지 16장 (값 범위 -0.5 ~ 0.5)
    enc1, dec1 = grad_norms(model, x, use_ste=True)
    enc0, dec0 = grad_norms(model, x, use_ste=False)
    print("층         | STE 있음   | STE 없음")
    print(f"Encoder    | {enc1:10.5f} | {enc0:10.5f}")
    print(f"Decoder    | {dec1:10.5f} | {dec0:10.5f}")
    print("\n(숫자 = 기울기 크기. 0이면 그 층은 전혀 학습되지 않음)")
    print("질문) STE가 없을 때 Encoder 기울기는 왜 0일까? 양자화 함수(계단 모양)의 기울기를 생각해 보자.")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print(f"아직 안 채운 함수가 있어요: {e}  -> check_week1.py 를 먼저 통과하세요")
