"""
Week 1 자가 채점: 내가 채운 함수들이 맞게 동작하는지 작은 예제로 확인한다.

    python check_week1.py

모든 항목이 [PASS]면 main.py로 학습해도 된다.
"""
import torch
import utils
from models.channel import index_to_bits, bits_to_index, bsc

CFG = dict(n_hiddens=128, n_residual_hiddens=32, n_residual_layers=2,
           n_embeddings=256, embedding_dim=8, beta=0.25, m=2)
results = []


def check(name, fn):
    # fn()이 True면 PASS. 아직 안 채운 TODO면 [TODO]로 표시
    try:
        ok = bool(fn())
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    except NotImplementedError as e:
        results.append(False)
        print(f"[TODO] {name}  <- {e}  (코드를 썼다면 raise 줄을 지웠는지 확인)")
    except NameError as e:
        results.append(False)
        print(f"[ERR ] {name}  <- {e}  (주석에 적힌 변수 이름 new_levels / bits / idx 를 그대로 썼는지 확인)")
    except Exception as e:
        results.append(False)
        print(f"[ERR ] {name}  <- {type(e).__name__}: {e}")


def quantizer(method, B):
    return utils.build_model(dict(CFG, method=method, B=B)).scalar_quantization


def close(a, b, tol=1e-3):
    return torch.allclose(torch.as_tensor(a).float(), torch.as_tensor(b).float(), atol=tol)


# ---------------- 1) Uniform ----------------
z = torch.tensor([-1.0, -0.6, -0.1, 0.2, 0.99, 1.0])
sq = quantizer('uniform', 2)
check("uniform_index  (B=2)",
      lambda: sq.uniform_index(z, 2).tolist() == [0, 0, 1, 2, 3, 3])
check("uniform_value  (B=2)",
      lambda: close(sq.uniform_value(torch.tensor([0, 1, 2, 3]), 2), [-0.75, -0.25, 0.25, 0.75]))
check("uniform_quantization 오차 <= step/2  (B=4)",
      lambda: (sq.uniform_quantization(torch.linspace(-1, 1, 1001), 4)
               - torch.linspace(-1, 1, 1001)).abs().max() <= 2 / 16 / 2 + 1e-6)

# ---------------- 2) mu-law ----------------
sq = quantizer('mulaw', 4)
check("mu_compress(0.01), mu=255 = 0.2285",
      lambda: close(sq.mu_compress(torch.tensor([0.0, 0.01, 1.0])), [0.0, 0.2285, 1.0]))
check("mu_expand(mu_compress(z)) == z",
      lambda: close(sq.mu_expand(sq.mu_compress(torch.linspace(-1, 1, 101))), torch.linspace(-1, 1, 101)))
check("mu-law 레벨은 0 근처가 더 촘촘",
      lambda: (lambda q: (q[8] - q[7]) < (q[15] - q[14]))(
          sq.mu_expand(sq.uniform_value(torch.arange(16), 4))))

# ---------------- 3) Lloyd-Max ----------------
sq = quantizer('lloydmax', 1)
samples = torch.tensor([-1.0, -0.8, 0.5, 0.7])
check("lloyd_max_step: 두 점 묶음의 평균으로 이동",
      lambda: close(sq.lloyd_max_step(samples, torch.tensor([-0.5, 0.5])), [-0.9, 0.6]))
sq = quantizer('lloydmax', 3)
g = torch.randn(20000).clamp(-3, 3) * 0.3
check("Lloyd-Max MSE < Uniform MSE  (가우시안 입력, B=3)",
      lambda: (sq.fit_lloyd_max(g) is not None) and
      ((sq.Loyld_max_quantization(g, 3) - g) ** 2).mean() < ((sq.uniform_quantization(g, 3) - g) ** 2).mean())

# ---------------- 4) Learned levels ----------------
sq = quantizer('learned', 2)
check("level_loss = codebook + beta * commitment",
      lambda: close(sq.level_loss(torch.tensor([0.0, 1.0]), torch.tensor([0.5, 0.5])), 0.25 * 1.25))

# ---------------- 5) 비트스트림 / BSC ----------------
check("index_to_bits(5, B=4) = [0,1,0,1]  (MSB 먼저)",
      lambda: index_to_bits(torch.tensor([5]), 4).tolist() == [0, 1, 0, 1])
idx = torch.randint(0, 16, (2, 8, 8, 8))
check("bits_to_index(index_to_bits(idx)) == idx",
      lambda: torch.equal(bits_to_index(index_to_bits(idx, 4), 4, idx.shape), idx))
bits = torch.randint(0, 2, (4096,), dtype=torch.uint8)
check("bsc: p=0 그대로, p=1 전부 반전",
      lambda: torch.equal(bsc(bits, 0.0), bits) and torch.equal(bsc(bits, 1.0), 1 - bits))
check("bsc: p=0.1 이면 약 10% 반전",
      lambda: 0.08 < (bsc(bits, 0.1) != bits).float().mean() < 0.12)

# ---------------- 6) 전체 모델 ----------------
x = torch.rand(2, 3, 32, 32) - 0.5
for method in ['uniform', 'mulaw', 'lloydmax', 'learned']:
    check(f"VQVAE forward ({method}, B=4)",
          lambda m=method: utils.build_model(dict(CFG, method=m, B=4))(x)[1].shape == x.shape)


def forward_on(device, p):
    # 모델과 입력을 device에 올리고, 채널(BSC p)까지 통과시켜 본다
    model = utils.build_model(dict(CFG, method='uniform', B=4)).to(device).eval()
    model.scalar_quantization.channel_p = p
    with torch.no_grad():
        return model(x.to(device))[1].shape == x.shape


check("VQVAE forward + BSC(p=0.1)  (CPU)", lambda: forward_on('cpu', 0.1))

# ---------------- 7) GPU (있을 때만) ----------------
# 새로 만드는 텐서(torch.rand, torch.arange 등)는 device=... 를 맞춰야 GPU에서도 돈다
if torch.cuda.is_available():
    check("VQVAE forward + BSC(p=0.1)  (GPU)", lambda: forward_on('cuda', 0.1))
else:
    print("[SKIP] GPU 검사 (CUDA GPU 없음)")

print(f"\n{sum(results)}/{len(results)} 통과")
if all(results):
    print("모두 통과! 이제 README의 다음 단계(학습)로 넘어가세요.")
