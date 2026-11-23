# 소리로 사진 보내기: Autoencoder 기반 이미지 통신

노트북 A가 사진을 압축해 소리로 보내고, 노트북 B가 마이크로 받아 사진을 복원합니다.

```
[노트북 A: 보내는 쪽]                              [노트북 B: 받는 쪽]
사진 -> 압축(Encoder) -> 0101... -> 소리  ~~~>  마이크 -> 0101... -> 복원(Decoder) -> 사진
```

| 순서 | 할 일 | 시간 | 언제 |
|---|---|---|---|
| [1](#1-설치-노트북마다-한-번만) | 설치 (코드, conda, 환경) | 30분 정도 | 노트북마다 한 번 |
| [2](#2-보낼-사진-고르기-보내는-쪽) | 보낼 사진 넣기 | 1분 | |
| [3](#3-보내고-받기) | 보내고 받기 | 1~5분 | 매번 |
| [4](#4-옵션을-바꿔-가며-실험하기) | 옵션 바꿔 가며 실험 | | |

**준비물**: 노트북 2대 (Windows / Mac 어떤 조합이든), 조용한 방.
노트북 한 대로는 안 됩니다. 노트북 마이크의 에코 제거 기능이 자기 스피커 소리를 지웁니다.

---

## 1. 설치 (노트북마다 한 번)

두 노트북 모두 합니다.

### 1-1. 코드 받기

1. **https://github.com/seonha01/WCML-EECE199/archive/refs/heads/main.zip** 에서 ZIP 을 받습니다.
2. 압축을 풀고 `WCML-EECE199-main` 폴더 이름을 `WCML-EECE199` 로 바꿔 옮깁니다.
   - **Windows**: `C:\WCML-EECE199` (경로에 한글, 띄어쓰기가 없게)
   - **Mac**: `~/WCML-EECE199` (홈 폴더)
3. 폴더 안에 `README.md`, `environment.yml`, `week1`, `week2` 가 있어야 합니다.

git: `git clone https://github.com/seonha01/WCML-EECE199.git` (새 버전은 `git pull`)

### 1-2. Miniforge(conda) 설치

이미 conda(Miniforge / Miniconda / Anaconda)가 있으면 건너뜁니다.

**Windows**
1. **https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe** 를 받아 실행합니다.
   ("Windows의 PC 보호" 창이 뜨면 **추가 정보 → 실행**)
2. **Next → I Agree → Just Me → Next → Install → Finish** (나머지는 기본값).
3. `Windows` 키 → `miniforge` → **Miniforge Prompt**. 줄 앞에 `(base)` 가 보이면 설치된 것입니다.

**Mac**
1. **터미널**을 엽니다 (`Cmd + Space` → `터미널`).
2. 아래 두 줄을 한 줄씩 실행합니다. 약관은 스페이스바로 넘기고 `yes`, 경로는 Enter, 마지막 질문도 `yes`.
   ```
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
3. 터미널을 껐다가 다시 열면 줄 앞에 `(base)` 가 보입니다.

### 1-3. 환경 만들기 (5~20분)

Windows 는 **Miniforge Prompt**, Mac 은 **터미널**에서:

| Windows | Mac |
|---|---|
| `cd /d C:\WCML-EECE199` | `cd ~/WCML-EECE199` |
| `conda env create -f environment.yml` | `conda env create -f environment.yml` |

마지막에 `conda activate eece101` 안내가 나오면 끝입니다.

### 1-4. 마이크 권한

- **Windows**: 설정 → 개인 정보 및 보안 → 마이크 → **"데스크톱 앱이 마이크에 액세스하도록 허용"** 켜기
- **Mac**: 처음 녹음할 때 뜨는 마이크 접근 창에서 **허용**.
  거부했다면 시스템 설정 → 개인정보 보호 및 보안 → 마이크 → **터미널** 켜기

---

## 2. 보낼 사진 고르기 (보내는 쪽)

`week2/images` 폴더에 사진을 넣습니다. 기본으로 `Lenna.png` 가 있습니다.

- 한 번에 한 장을 보냅니다. 사진이 여러 장이면 보낼 때 번호로 고릅니다.
- 정사각형이 아닌 사진은 비율에 맞춰 조각을 나눕니다 (예: 가로로 긴 사진을 `--tiles 3` 으로 보내면 가로 3 x 세로 2 조각).
  비율을 맞추느라 가장자리가 조금 잘릴 수 있습니다.
- 큰 대상 하나가 가까이 찍힌 사진(얼굴, 동물, 과일)이 잘 나옵니다. 글씨나 잔무늬는 뭉개집니다.

보내는 쪽은 사진의 지문(파일 내용으로 만든 숫자)과 조각 배치를 소리 맨 앞에 같이 보냅니다.
받는 쪽은 자기 `images` 폴더에서 지문이 같은 사진을 찾으면 원본과 비교한 점수(PSNR)까지 보여 주고, 없으면 받은 사진만 보여 줍니다.
원본과 비교하려면 같은 사진 파일을 양쪽 `images` 폴더에 넣어 둡니다 (이름은 달라도 됨).

---

## 3. 보내고 받기

받는 쪽을 먼저 실행하고, `녹음 시작` 이 나오면 보내는 쪽을 실행합니다.

### 폴더로 이동 (두 노트북 모두)

| Windows (Miniforge Prompt) | Mac (터미널) |
|---|---|
| `cd /d C:\WCML-EECE199\week2` | `cd ~/WCML-EECE199/week2` |

### 실행

| | Windows | Mac |
|---|---|---|
| ① 받는 쪽 (먼저) | `scripts_windows_rx.bat` | `bash scripts_mac_rx.sh` |
| ② 보내는 쪽 | `scripts_windows_tx.bat` | `bash scripts_mac_tx.sh` |

- 보내는 쪽은 3초를 센 뒤 소리를 냅니다 (기본 설정에서 약 30초).
- 받는 쪽은 소리 길이에 맞춰 녹음하고, 받은 사진을 `week2/results/rx_result.png` 로 저장해 엽니다.
  보낸 사진이 받는 쪽 `images` 에 있으면 [원본 | 압축만 했을 때 | 받은 것] 세 장을 나란히 보여 줍니다.
- Windows 는 탐색기에서 `.bat` 을 더블클릭해도 됩니다. PowerShell 에서는 앞에 `.\` 를 붙입니다 (`.\scripts_windows_rx.bat`).

### 결과 읽는 법

```
프레임 시작(chirp) 찾음 8번  ->  헤더까지 읽은 프레임 8/8  (손실률 0.0%),  CRC 실패 0개
BER (수신된 비트 기준) 0.00e+00
PSNR  채널 없음 24.44 dB  ->  수신 이미지 24.44 dB
```

| 항목 | 뜻 | 좋은 값 |
|---|---|---|
| 프레임 8/8 | 데이터를 묶음(프레임) 8개로 나눠 보냈고 그중 몇 개를 받았나. 못 받은 자리는 회색 | 전부 |
| CRC 실패 | 받았지만 틀린 비트가 있는 묶음 수 | 0 |
| BER | 틀린 비트 비율 (`1e-02` = 100개 중 1개) | 0 |
| PSNR | 원본과 비슷한 정도 (dB, 클수록 좋음). "채널 없음" = 압축만 했을 때 = 최고점 | 두 숫자가 같음 |

`[자동]` 은 소리에서 알아낸 보내는 쪽 설정, `[진단]` 은 문제의 원인과 해결 방법입니다.

---

## 4. 옵션을 바꿔 가며 실험하기

명령 뒤에 붙입니다. 첫 숫자는 압축률 B 입니다.

```
scripts_windows_tx.bat 8 --tiles 2          (Windows)
bash scripts_mac_tx.sh 8 --tiles 2          (Mac)
```

받는 쪽은 B, 조각 배치, 보낸 사진을 소리에서 알아냅니다. 녹음 시간은 B 와 tiles 로 정해지므로 양쪽에 같은 숫자와 `--tiles` 를 줍니다.

| 옵션 | 뜻 | 값 |
|---|---|---|
| 첫 숫자 (B) | 압축률. 숫자 하나를 몇 비트로 보낼지 | `2` (1/24) / `4` (1/12, 기본) / `8` (1/6) |
| `--tiles N` | 해상도. 사진의 긴 쪽을 32x32 조각 N 개로 나눠 보냄 | `1` (32x32, 기본) ~ `4` (128x128) |
| `--image 이름` | 보낼 사진을 바로 지정 (images 폴더 안) | `Lenna.png` |
| `--baud 200` | 2배 빠르게 (방이 울리면 오류가 늘어남) | 기본 `100` |
| `--interleave` | 묶음을 잃어도 손상이 한 곳에 몰리지 않고 흩어짐 (양쪽 같게) | |

### 자주 쓰는 명령 (받는 쪽 먼저, 양쪽 같은 숫자와 옵션)

| 해상도 | 받는 쪽 | 보내는 쪽 | 시간 (B=4) |
|---|---|---|---|
| 32x32 | `bash scripts_mac_rx.sh 4` | `scripts_windows_tx.bat 4` | 32초 |
| 64x64 | `bash scripts_mac_rx.sh 4 --tiles 2` | `scripts_windows_tx.bat 4 --tiles 2` | 2분 |
| 96x96 | `bash scripts_mac_rx.sh 4 --tiles 3` | `scripts_windows_tx.bat 4 --tiles 3` | 4분 40초 |
| 96x96 (빠르게) | `bash scripts_mac_rx.sh 2 --tiles 3` | `scripts_windows_tx.bat 2 --tiles 3` | 2분 20초 |

받는 쪽과 보내는 쪽이 반대면 `scripts_windows_rx.bat`, `bash scripts_mac_tx.sh` 로 바꿉니다 (뒤 옵션은 같음).

### 보내는 시간 (기본 100 baud, 정사각형 사진)

| | B = 2 | B = 4 | B = 8 |
|---|---|---|---|
| tiles 1 (32x32) | 17초 | 32초 | 1분 |
| tiles 2 (64x64) | 1분 | 2분 | 4분 |
| tiles 3 (96x96) | 2분 20초 | 4분 40초 | 9분 |
| tiles 4 (128x128) | 4분 | 8분 | 불가 (프레임 255개 초과) |

### 실험 예시

1. **압축률**: 같은 사진을 B = 2, 4, 8 로 보내고 화질과 시간 비교
2. **해상도**: `--tiles 1` 과 `--tiles 2` 비교. 조각 경계가 보이는가
3. **속도와 오류**: 양쪽 `--baud 200`. CRC 실패와 BER 이 어떻게 변하고, 왜 그런가
4. **거리와 소리 크기**: 노트북을 멀리, 볼륨을 작게 하면 어디서부터 깨지고, 사진은 어떻게 망가지는가
5. **인터리빙**: 소리 중간에 마이크를 잠깐 막고, `--interleave` 를 켰을 때와 비교

작동 원리, 예외 상황 처리, 모든 옵션, 시뮬레이션(스피커 없이 컴퓨터 안에서 실험)은 [`week2/README.md`](week2/README.md) 에 있습니다.

---

## 자주 나오는 문제

| 증상 | 해결 |
|---|---|
| `'conda'은(는) 내부 또는 외부 명령...이 아닙니다` | 일반 명령 프롬프트 대신 **Miniforge Prompt** 에서 실행 |
| `[error] conda env 'eece101' not found` | 1-3 (환경 만들기) 실행 |
| `'scripts_windows_rx.bat' is not recognized` | `week2` 폴더에서 실행. PowerShell 이면 앞에 `.\` |
| `No such file or directory: scripts_mac_rx.sh` | `cd ~/WCML-EECE199/week2` |
| `chirp 를 하나도 못 찾았습니다` | 소리가 마이크에 안 들어옴. 볼륨 올리기, 30cm 이내로, 마이크 권한(1-4) 확인 |
| `CRC 실패` 가 많고 사진에 얼룩 | 더 가깝게, 조용하게, 스피커를 상대 마이크 쪽으로. `--baud 200` 을 썼다면 빼기 |
| 사진 일부가 회색 | 그 묶음을 못 받음. 위와 같음 |
| `앞부분 프레임 ...개를 놓쳤습니다` | 받는 쪽을 먼저 시작하고 `녹음 시작` 을 본 뒤 보내기 |
| `뒷부분을 놓쳤습니다` | 받는 쪽에 보내는 쪽과 같은 B / `--tiles` 를 주기 (녹음 시간이 그걸로 정해짐) |
| `보낸 사진이 이 컴퓨터의 images 폴더에 없음` | 받은 사진만 표시. 원본 비교는 같은 사진 파일을 받는 쪽 `images` 에도 넣기 |
| `conda env create` 가 `prefix already exists` | 이미 있음. 그대로 사용 |
| 환경 만들다 실패 | `conda env remove -n eece101` 후 다시 `conda env create -f environment.yml` |
| 다른 마이크를 쓰고 싶음 | `python -m sounddevice` 로 번호 확인 후 받는 쪽 명령 뒤에 `--device 번호` |
