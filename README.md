# 연구참여: Autoencoder 기반 이미지 통신

```
이미지 -> Encoder -> latent z -> 양자화(B bit) -> 비트열 -> [채널] -> Decoder -> 복원 이미지
```

**컴퓨터에 아무것도 설치되어 있지 않다고 가정하고** 처음부터 설명합니다.
순서대로 따라 하면 30분 정도 걸립니다 (대부분 다운로드 기다리는 시간).
Windows 기준이고, Mac은 각 단계의 **[Mac]** 부분을 보세요.

| 단계 | 할 일 | 한 번만? |
|---|---|---|
| 1 | 코드 받기 (ZIP) | 과제가 올라올 때마다 |
| 2 | Miniforge(conda) 설치 | 한 번만 |
| 3 | 과제용 환경 만들기 | 한 번만 |
| 4 | 과제 시작 | 매번 |

---

## 1단계. 코드 받기

1. 아래 링크를 누르면 `EECE101-main.zip` 파일이 다운로드됩니다.

   **https://github.com/seonha01/EECE101/archive/refs/heads/main.zip**

   (또는 GitHub 페이지 위쪽의 초록색 **`<> Code`** 버튼 → **Download ZIP**)
2. 다운로드 폴더에서 ZIP 파일을 **마우스 오른쪽 클릭 → 압축 풀기(모두 추출)**.
3. 생긴 `EECE101-main` 폴더를 **`C:\` 바로 아래로 옮기세요.** → `C:\EECE101-main`
   - 경로에 한글이나 띄어쓰기가 있으면 가끔 오류가 나서, 짧고 영어로만 된 곳에 두는 게 안전합니다.
   - 폴더를 열었을 때 `README.md`, `environment.yml`, `week1` 이 바로 보여야 합니다.
     (`EECE101-main` 안에 `EECE101-main`이 또 있으면 안쪽 폴더를 옮기세요)

**[Mac]** ZIP을 더블클릭하면 풀립니다. `EECE101-main` 폴더를 홈 폴더(집 모양 아이콘)로 옮기세요.

> 과제가 새로 올라오면 ZIP을 다시 받아서 **다른 이름의 폴더에** 푸세요 (내가 작성한 코드가 덮어써지지 않게).
> git을 아는 사람은 `git clone https://github.com/seonha01/EECE101.git` 로 받아도 됩니다.

---

## 2단계. Miniforge 설치 (conda)

**conda**는 파이썬과 라이브러리(PyTorch 등)를 과제별로 따로 설치해 주는 프로그램이고,
**Miniforge**는 conda를 설치하는 가장 가벼운 방법입니다. 파이썬이 없어도 같이 설치됩니다.

### Windows

1. 아래 링크를 누르면 설치 파일(`Miniforge3-Windows-x86_64.exe`, 약 80MB)이 다운로드됩니다.

   **https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe**

   (링크가 안 되면 https://github.com/conda-forge/miniforge 페이지의 *Download* 표에서 `Windows x86_64` 줄의 파일)
2. 다운로드한 파일을 더블클릭해서 실행합니다.
   - "Windows의 PC 보호" 파란 창이 뜨면 **추가 정보 → 실행**
3. 설치 화면은 이렇게 누르면 됩니다.

   | 화면 | 누를 것 |
   |---|---|
   | Welcome | **Next** |
   | License Agreement | **I Agree** |
   | Select Installation Type | **Just Me (recommended)** 선택 → **Next** |
   | Choose Install Location | 그대로 두고 **Next** (경로에 한글이 있다는 경고가 나오면 `C:\miniforge3`로 바꾸기) |
   | Advanced Installation Options | 체크 상태 **그대로** 두고 **Install** |
   | Completed | **Next** → **Finish** |

4. **확인**: 키보드 `Windows` 키 → `miniforge` 입력 → **Miniforge Prompt** 클릭.
   검은 창이 열리고 줄 맨 앞에 `(base)`가 보이면 성공입니다.

   ```
   (base) C:\Users\내이름>
   ```
   여기에 `conda --version` 을 입력하고 Enter → `conda 26.x.x` 같은 버전 숫자가 나오면 OK.

> **앞으로 모든 명령은 이 "Miniforge Prompt" 창에 입력합니다.**
> 일반 "명령 프롬프트"나 "PowerShell"에서는 `conda`를 못 찾을 수 있어요.

### [Mac]

1. **터미널**을 엽니다 (`Cmd + Space` → `터미널` 또는 `Terminal` 입력 → Enter).
2. 아래 두 줄을 한 줄씩 복사해서 붙여넣고 Enter. (중간에 약관이 나오면 스페이스바로 넘기고 `yes`, 경로는 Enter,
   마지막 "initialize" 질문에도 `yes`)

   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```
3. 터미널을 **완전히 종료했다가 다시 열면** 줄 앞에 `(base)`가 보입니다. `conda --version`으로 확인.

---

## 3단계. 과제용 환경 만들기

과제에 필요한 파이썬 3.10과 PyTorch, numpy 등을 `eece101`이라는 이름의 환경에 설치합니다.

1. Miniforge Prompt에서 코드 폴더로 이동합니다.

   ```
   cd /d C:\EECE101-main
   ```
   - `cd`는 "폴더 이동" 명령입니다. 폴더 경로를 모르겠으면 탐색기에서 그 폴더를 열고
     위쪽 주소창을 클릭 → 경로 복사 → `cd /d ` 뒤에 붙여넣기 (오른쪽 클릭하면 붙여넣어짐).
   - `dir` 을 입력했을 때 `environment.yml`이 보이면 맞는 폴더입니다.
   - **[Mac]** `cd ~/EECE101-main` , 파일 목록은 `ls`
2. 환경을 만듭니다. **인터넷 속도에 따라 5~20분** 걸립니다. 글자가 계속 올라가면 정상이니 기다리세요.

   ```
   conda env create -f environment.yml
   ```
   마지막에 `conda activate eece101` 하라는 안내가 나오면 성공입니다.
3. 환경을 켭니다.

   ```
   conda activate eece101
   ```
   줄 맨 앞이 `(base)`에서 **`(eece101)`** 로 바뀌면 성공입니다.
4. **확인**:

   ```
   python -c "import torch; print('PyTorch', torch.__version__)"
   ```
   `PyTorch 2.x.x` 가 나오면 준비 끝!

> **Miniforge Prompt를 새로 열 때마다** `conda activate eece101` 을 먼저 입력해야 합니다.
> (줄 맨 앞에 `(eece101)`이 있는지 항상 확인)

---

## 4단계. 과제 시작

```
conda activate eece101
cd /d C:\EECE101-main\week1
python check_week1.py
```

처음엔 전부 `[TODO]`가 나옵니다. 이제 **`week1` 폴더 안의 `README.md`** 를 열고 순서대로 따라가세요.

- 코드 편집은 메모장도 되지만 **VS Code**(https://code.visualstudio.com/, 무료)를 추천합니다.
  설치 후 **File → Open Folder** 로 `C:\EECE101-main` 을 열면 왼쪽에 파일 목록이 보입니다.
  `.md` 파일은 오른쪽 위 미리보기 버튼(돋보기 달린 아이콘)을 누르면 보기 좋게 보입니다.
- 명령 실행은 계속 **Miniforge Prompt**에서 하면 됩니다.

### 데이터 (CIFAR-100)

처음 실행할 때 `C:\EECE101-main\data` 폴더에 **자동으로 다운로드**됩니다 (약 170MB).
서버가 느려서 오래 걸리거나 중간에 끊길 수 있습니다. 조교에게 `data` 폴더를 받았다면
`C:\EECE101-main\data` 위치에 넣으세요.

---

## CPU / GPU

**CPU만 있어도 과제를 전부 할 수 있습니다.** (1주차 모델 1개 학습이 노트북 CPU로 약 4~10분)

NVIDIA GPU가 있으면 자동으로 GPU를 씁니다 (실행하면 처음에 `device: cuda` 또는 `device: cpu`가 출력됨).

```
python main.py --method uniform --B 4                 # 자동 선택
python main.py --method uniform --B 4 --device cpu    # GPU가 있어도 CPU로
```

- GPU를 쓰려면 CUDA용 PyTorch가 필요합니다: `environment.yml`의 `whl/cpu`를
  [pytorch.org](https://pytorch.org/get-started/locally/)에서 본인 CUDA에 맞는 주소(`whl/cu1xx`)로 바꾼 뒤 3단계를 진행.
  확인: `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- GPU에서 학습한 모델도 CPU에서 그대로 불러올 수 있고, 그 반대도 됩니다.

---

## 자주 나오는 문제

| 증상 | 해결 |
|---|---|
| `'conda'은(는) 내부 또는 외부 명령...이 아닙니다` | 일반 명령 프롬프트가 아니라 **Miniforge Prompt**에서 실행 |
| `No module named torch` | `conda activate eece101`을 안 했음 (줄 맨 앞에 `(eece101)`이 있어야 함) |
| `EnvironmentFileNotFound` / `environment.yml` 없음 | 폴더 위치가 다름. `cd /d C:\EECE101-main` 후 `dir`로 확인 |
| `can't open file 'check_week1.py'` | `cd week1` 로 과제 폴더까지 들어가야 함 |
| `conda env create`가 `prefix already exists`로 실패 | 이미 만들어져 있음. `conda activate eece101`만 하면 됨 |
| 환경 만들다 중간에 멈춤/실패 | `conda env remove -n eece101` 후 다시 `conda env create -f environment.yml` |
| CIFAR-100 다운로드가 멈추거나 `File not found or corrupted` | `data` 폴더를 지우고 다시 실행하거나, 조교에게 `data` 폴더 받기 |
| Miniconda/Anaconda를 쓰는데 "Terms of Service" 오류 | 오류 메시지의 `conda tos accept ...` 명령을 실행해 약관에 동의하거나, Miniforge로 설치 |
| `CUDA GPU를 찾을 수 없습니다` | `--device cpu`를 붙이거나 빼기 (기본값 auto면 알아서 CPU 사용) |
