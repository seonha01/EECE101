# Week 1 과제: 스칼라 양자화 (Scalar Quantization)

```
이미지(3x32x32) -> Encoder -> tanh -> z (8x8x8, [-1,1])
      -> 양자화 인덱스 (B bit)  -> [비트스트림 -> BSC(p) -> 비트스트림] -> z_q -> Decoder -> 복원 이미지
```

Encoder / Decoder / 학습 코드는 다 되어 있습니다. **`TODO` 표시된 빈칸만** 채우면 됩니다.
빈칸은 전부 1~3줄이고, 주석에 힌트와 예시 답이 있습니다.

## 채울 곳

| 번호 | 파일 | 함수 | 내용 |
|---|---|---|---|
| 1-1 | `models/scalarquantizer.py` | `uniform_index` | z → 칸 번호 |
| 1-2 | 〃 | `uniform_value` | 칸 번호 → 칸 가운데 값 |
| 2-1 | 〃 | `mu_compress` | μ-law 압축 F(z) |
| 2-2 | 〃 | `mu_expand` | μ-law 신장 F⁻¹(y) |
| 3   | 〃 | `lloyd_max_step` | Lloyd 알고리즘: 구간별 평균으로 레벨 갱신 |
| 4   | 〃 | `level_loss` | 학습 가능한 레벨의 loss (codebook + β·commitment) |
| 5-1 | `models/channel.py` | `index_to_bits` | 정수 → B비트 |
| 5-2 | 〃 | `bits_to_index` | B비트 → 정수 |
| 5-3 | 〃 | `bsc` | 비트를 확률 p로 뒤집기 |

5번(채널)은 1~4를 끝낸 뒤 해도 됩니다. 학습(`main.py`)은 채널 없이(p=0) 하므로 5번이 없어도 되고,
`week1_experiment.py`의 p > 0 실험에서만 5번이 필요합니다.

## 순서

`week1` 폴더에서 (`cd week1`):

```powershell
# 1) 빈칸을 채울 때마다 자가 채점 -> [TODO] 가 [PASS] 로 바뀌는지 확인
python check_week1.py

# 2) 17/17 통과하면 학습 (CPU 기준 모델당 약 4~10분)
python main.py --method uniform  --B 4 --n_updates 3000
python main.py --method mulaw    --B 4 --n_updates 3000
python main.py --method lloydmax --B 4 --n_updates 3000
python main.py --method learned  --B 4 --n_updates 3000

# 3) PSNR 표 + 복원 이미지 그림 (checkpoints/ 에 있는 모델만 평가)
python week1_experiment.py --Bs 4
```

결과는 `results/table.md`, `results/psnr.png`, `results/recon_*.png`에 저장됩니다.
시간이 있으면 B=2, 8도 학습해서 `python week1_experiment.py` (B ∈ {2,4,8}, p ∈ {0, 0.01, 0.1})로 비교해 보세요.

## 생각해 볼 것 (보고서)

1. **전송 비트 수**: latent가 8×8×8일 때 이미지 한 장당 몇 bit를 보내나? 원본(3×32×32×8 bit) 대비 몇 배 압축인가?
2. **`z_q = z + (z_q - z).detach()`** (`forward` 안): 무슨 뜻이고, 무슨 역할을 하며, 왜 필요할까?
   (힌트: 양자화 함수의 미분값은 거의 모든 곳에서 0이다. 이 줄을 `z_q = z_q`로 바꾸면 Encoder는 학습될까?)
3. 네 가지 양자화기의 PSNR을 B별로 비교하고 이유를 설명하기. μ-law의 레벨은 어디에 몰려 있나?
4. BSC(p)에서 B를 늘리면 PSNR이 어떻게 되나? 비트를 늘리는 게 항상 좋은가?
