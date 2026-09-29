# Week 1 과제: 스칼라 양자화 (Scalar Quantization)

```
이미지(3x32x32) -> Encoder -> tanh -> z (8x8x8, [-1,1])
      -> 양자화 인덱스 (B bit)  -> [비트스트림 -> BSC(p) -> 비트스트림] -> z_q -> Decoder -> 복원 이미지
```

Encoder / Decoder / 학습 코드는 다 되어 있습니다. **`TODO` 표시된 빈칸만** 채우면 됩니다.
빈칸은 전부 1~3줄이고, 주석에 힌트와 예시 답이 있습니다.

## 예상 시간

| 할 일 | 시간 |
|---|---|
| 빈칸 9개 채우기 | 30~90분 |
| 모델 학습 6개 (컴퓨터가 하는 동안 다른 일 가능) | CPU 기준 약 30~60분 |
| 실험 + 그림 | 약 10분 |
| 보고서 | 1~3시간 |

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

> **채우는 법**: 각 빈칸의 `raise NotImplementedError(...)` 줄을 **지우고** 그 자리에 코드를 씁니다.
> raise 줄이 남아 있으면 위에 코드를 써도 계속 `[TODO]`로 나옵니다.
> 3번, 5-1번, 5-2번은 주석에 적힌 **변수 이름**(`new_levels`, `bits`, `idx`)을 그대로 써야 합니다.

## 순서

Miniforge Prompt에서 (`conda activate eece101` 후) `week1` 폴더로 이동:

```
cd /d C:\EECE101-main\week1
```

### 1) 빈칸 채우기

빈칸을 하나 채울 때마다 자가 채점을 돌려서 `[TODO]` → `[PASS]`로 바뀌는지 확인합니다.

```
python check_week1.py
```

마지막 줄에 **"모두 통과!"** 가 나오면 다음으로. (GPU가 없으면 `[SKIP] GPU 검사`는 정상입니다)

### 2) 모델 학습 (6개)

모델 하나에 CPU로 약 4~10분. 한 줄씩 실행해도 되고, 아래처럼 한 번에 돌려 놓고 기다려도 됩니다.

```
for %m in (uniform mulaw lloydmax learned) do python main.py --method %m --B 4
python main.py --method uniform --B 2
python main.py --method uniform --B 8
```

- 앞의 4개(B=4)는 보고서 3번, 뒤의 2개(uniform B=2, 8)는 보고서 4번에 씁니다.
- 학습한 모델은 `checkpoints/` 폴더에 저장됩니다. 중간에 멈췄으면 그 모델만 다시 실행하면 됩니다.
- [Mac] 첫 줄 대신: `for m in uniform mulaw lloydmax learned; do python main.py --method $m --B 4; done`

### 3) 실험 + 그림

```
python week1_experiment.py
python show_levels.py
python check_ste.py
```

| 명령 | 결과 | 보고서 |
|---|---|---|
| `week1_experiment.py` | `results/table.md` (PSNR 표), `results/psnr.png`, `results/recon_*.png` (복원 이미지) | 3, 4번 |
| `show_levels.py` | `results/levels_B4.png`: 방법별 레벨이 z 분포의 어디에 있는지 | 3번 |
| `check_ste.py` | STE 있을 때 / 없을 때 기울기 크기 (화면 출력) | 2번 |

`week1_experiment.py`는 `checkpoints/`에 있는 모델만 평가하고, 없는 조합은 `[skip]`으로 건너뜁니다 (정상).

## 보고서

1. **전송 비트 수**: latent가 8×8×8일 때 이미지 한 장당 몇 bit를 보내나? 원본(3×32×32×8 bit) 대비 몇 배 압축인가?
   B = 2, 4, 8 각각 계산하기.
2. **`z_q = z + (z_q - z).detach()`** (`scalarquantizer.py`의 `forward` 안)
   - `python check_ste.py` 결과 표를 붙이고, STE가 없을 때 Encoder 기울기가 어떻게 되는지 쓰기.
   - 그러면 STE 없이 학습하면 Encoder는 어떻게 될까? 이 한 줄이 왜 필요한지 결과를 근거로 설명하기.
3. **B=4에서 네 가지 양자화기 비교**
   - `results/table.md`에서 p=0일 때 네 방법의 PSNR을 비교하기.
   - `results/levels_B4.png`를 보고 방법마다 레벨이 어디에 몰려 있는지, z 분포(회색)와 잘 맞는지 설명하기.
     특히 μ-law 레벨은 어디에 몰려 있고, 그게 PSNR에 어떤 영향을 줬을까?
4. **비트 수 B와 채널 에러 (uniform)**
   - `results/table.md`에서 uniform의 B=2, 4, 8 × p=0, 0.01, 0.1 (9칸) PSNR을 표로 정리하기.
   - p=0일 때 B를 늘리면? p=0.1일 때 B를 늘리면? 비트를 늘리는 게 항상 좋은가?
   - `results/recon_uniform.png`에서 비트 에러가 이미지를 어떻게 망가뜨리는지 보기.
