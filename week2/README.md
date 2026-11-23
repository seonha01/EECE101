# week2: 소리로 사진 보내기, 작동 원리와 옵션

실습 방법(설치, 보내고 받기)은 [맨 위 README](../README.md) 에 있습니다. 여기는 **어떻게 동작하는지**와 **모든 옵션**입니다.

## 전체 흐름

```
보내는 쪽 (tx.py)
  사진 -> 32x32 조각들 -> Encoder -> 숫자 512개/조각 -> 양자화(B 비트) -> 비트열
       -> [사진 정보 x3 | 사진 비트] -> 256비트씩 묶음(프레임) -> [chirp | 헤더 x3 | 데이터 | CRC] -> BFSK 소리

받는 쪽 (rx.py)
  마이크 -> chirp 찾기(동기화) -> 헤더 읽기(다수결) -> 비트 복원 -> CRC 확인
       -> 사진 정보(지문, 조각 배치) -> 숫자 -> Decoder -> 조각 붙이기 -> 사진
       -> images 폴더에서 지문이 같은 사진을 찾으면 원본과 비교
```

| 단계 | 하는 일 | 코드 |
|---|---|---|
| 압축 | 1주차에 학습한 Autoencoder. 32x32 사진 하나 -> 숫자 512개 (8x8 위치 x 8채널) -> 숫자마다 B 비트 | `image_codec.py`, `week1/` |
| BFSK 변조 | 비트 0 = 3 kHz 소리, 비트 1 = 5 kHz 소리. 비트 하나 = 10 ms (100 baud) | `modem.fsk_modulate` |
| 동기화 | 묶음마다 앞에 1 -> 8 kHz 로 올라가는 "뿌우" 소리(chirp)를 붙이고, 받는 쪽은 녹음에서 이 소리를 찾아 시작 위치를 맞춤 | `modem.find_next_preamble` |
| 헤더 | 묶음 번호, 전체 묶음 수, 길이, B (32비트). 깨지면 묶음 전체를 못 쓰므로 3번 보내고 비트마다 다수결 | `modem.majority_header` |
| 복조 | 비트 구간마다 3 kHz 와 5 kHz 중 어느 쪽 소리가 큰지 비교 | `modem.fsk_demodulate` |
| CRC | 묶음이 온전히 왔는지 검사하는 16비트 값. **틀려도 버리지 않고** 사진 복원에 씀 (Autoencoder 는 비트가 조금 틀려도 괜찮음) | `modem.crc16` |
| 사진 정보 | 맨 앞 40비트 x 3번: 사진 지문(파일 내용의 SHA-1 앞 32비트) + 조각 배치(세로, 가로 각 4비트). 받는 쪽이 사진 크기와 원본을 알아냄 | `image_codec.info_bits` |
| 조각 | 모델이 32x32 만 다루므로 사진을 32x32 조각으로 나눠 각각 압축. 긴 쪽이 `--tiles` 개, 짧은 쪽은 비율대로 | `image_codec.to_patches` |

## 파일

| 파일 | 내용 |
|---|---|
| `tx.py` | 보내는 쪽. 맨 위 `B`, `TILES`, `IMAGE` 를 고쳐서 `python tx.py` 로 실행해도 됨 |
| `rx.py` | 받는 쪽. 녹음 -> 복원 -> 결과 출력 / 그림 (`results/rx_result.png`), 녹음은 `results/rx_record.wav` |
| `scripts_windows_tx.bat`, `scripts_windows_rx.bat` | Windows 실행 스크립트 (conda 환경을 알아서 켜고, 기본 `--baud 100 --gray`) |
| `scripts_mac_tx.sh`, `scripts_mac_rx.sh` | Mac 실행 스크립트 (같은 기능) |
| `modem.py` | 소리 <-> 비트 (BFSK, chirp, 헤더, CRC, 채널 시뮬레이션) |
| `image_codec.py` | 사진 <-> 비트 (1주차 모델, 조각 나누기/붙이기, 인터리빙) |
| `loopback.py` | 스피커 없이 컴퓨터 안에서 잡음/메아리/시계 오차를 흉내 내어 실험 |
| `explore_week2.py`, `check_week2.py` | 모뎀 각 단계를 그림으로 보기 / 단계별 자동 검사 (공부용) |
| `images/` | 보낼 사진 |

## 모든 옵션

스크립트(`scripts_*`)는 첫 숫자가 B 이고 나머지 옵션은 그대로 `tx.py` / `rx.py` 에 넘깁니다.
`python tx.py ...`, `python rx.py ...` 로 직접 실행할 때는 기본값이 `--baud 200` 이고 `--gray` 가 꺼져 있으니 양쪽을 맞추세요.

**사진 / 압축** (B, 조각 배치, 보낸 사진은 받는 쪽이 소리에서 알아냄. 녹음 시간 때문에 B 와 tiles 는 양쪽 같게)

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--B` | 4 | 숫자 하나당 비트 수 = 압축률. 2: 1/24, 4: 1/12, 8: 1/6 |
| `--tiles` | 1 | 사진의 긴 쪽을 32x32 조각 tiles 개로 (짧은 쪽은 비율대로, 최대 15) |
| `--method` | uniform | 1주차 양자화 방법: `uniform`, `mulaw`, `lloydmax`, `learned` |
| `--image` | | 사진 파일. 이름만 주면 `images/` 에서 찾음. tx 에서 안 주면 `images/` 에서 번호로 고르기. rx 에서 안 주면 지문으로 자동 |
| `--index` | | `--image` 대신 CIFAR-100 테스트 사진 번호 (데이터 필요, 아래 시뮬레이션 참고) |
| `--gray` | 끔 | Gray 코드: 비트 하나가 틀려도 숫자가 조금만 바뀜 -> 얼룩이 덜 튐 |
| `--interleave` | 끔 | 숫자 순서를 섞어 보냄 -> 묶음을 잃어도 손상이 사진 전체에 얇게 흩어짐 |

**소리** (양쪽 같게)

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--baud` | 200 (스크립트는 100) | 초당 비트 수. 크면 빠르지만 방 울림(메아리)에 약함 |
| `--f0`, `--f1` | 3000, 5000 | 비트 0 / 1 의 소리 높이 (Hz) |
| `--frame_bits` | 256 | 묶음 하나의 데이터 비트 수 |
| `--amp` | 0.5 | 소리 크기 (0~1) |
| `--sync_thr` | 0.25 | chirp 를 찾았다고 볼 닮음 정도 (0~1). 못 찾으면 낮춰 보기 |

**실행**

| 옵션 | 쪽 | 뜻 |
|---|---|---|
| `--out 파일.wav` | tx | 재생 대신 wav 로 저장 (휴대폰으로 옮겨 재생해도 됨) |
| `--wait` | tx | 재생 전 카운트다운 (초, 기본 3) |
| `--record 초` | rx | 녹음 시간. 안 주면 옵션으로 계산한 소리 길이 + 12초 |
| `--wav 파일.wav` | rx | 녹음 대신 wav 파일을 복원 (예: `results/rx_record.wav` 다시 분석) |
| `--device 번호` | 둘 다 | 스피커 / 마이크 선택 (`python -m sounddevice` 로 번호 확인) |
| `--no_show` | rx | 결과 그림을 자동으로 열지 않음 |

## 실측 결과 (Windows 노트북 -> MacBook Air, 약 30 cm, B=4, Lenna)

| 설정 | 받은 묶음 | CRC 실패 | BER | PSNR (최고점) |
|---|---|---|---|---|
| tiles 2, 200 baud | 25 / 32 | 27 | 7.9 % | 16.1 dB (26.3) |
| tiles 2, 100 baud + gray | 32 / 32 | 0 | 0 | **26.3 dB (26.3)** |

200 baud 는 비트 하나가 5 ms 라서, 방 벽에 반사된 앞 비트의 소리가 다음 비트를 덮어 오류가 많았던 것으로 보입니다.
비트를 10 ms 로 늘리자 오류가 사라졌습니다. 그래서 스크립트 기본값이 100 baud 입니다.

## 시뮬레이션 (스피커 없이)

컴퓨터 안에서 잡음, 메아리, 두 노트북의 시계 차이를 흉내 냅니다. CIFAR-100 테스트 사진을 쓰므로 데이터가 필요합니다
(`python ../download_data.py`, 약 170MB. 직접 고른 사진이면 `--image Lenna.png` 를 붙여 데이터 없이도 됨).

```
python loopback.py --image Lenna.png                         # 잡음 크기(SNR) 10, 0, -5, -8, -10 dB
python loopback.py --image Lenna.png --snr -8 -9 -10 -11     # 깨지기 시작하는 구간
python loopback.py --image Lenna.png --interleave --snr -10  # 묶음을 잃을 때 인터리빙 효과
python loopback.py --image Lenna.png --ppm 500 2000          # 시계 차이를 키우면?
python loopback.py --image Lenna.png --tiles 2 --B 8         # 해상도 / 압축률
```

결과 그림은 `results/loopback_snr*.png` ([원본 | 채널 없음 | 받은 것]).
SNR 은 신호와 잡음의 세기 비 (0 dB = 잡음이 신호만큼 큼, -10 dB = 잡음이 10배)입니다.

모뎀 각 단계를 그림으로 보려면 `python explore_week2.py 1` ~ `5` (1: 변조, 2: 복조, 3: 동기화, 4: 헤더 다수결, 5: 프레임).
