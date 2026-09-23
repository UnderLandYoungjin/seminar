# YOLOv8 비전 검사 따라하기 세미나

USB 카메라 한 대로 이미지를 찍고, 직접 라벨링해서, YOLOv8 모델을 학습시킨 다음, 실시간으로 OK/NG를 판정하는 데까지 한 번에 진행합니다.

> **용어 먼저 짚고 가기**
> - **YOLO** (You Only Look Once): 이미지를 한 번만 보고 물체의 위치(박스)와 종류(클래스)를 동시에 찾아내는 딥러닝 검출 모델입니다. 오늘 쓰는 건 v8 버전의 가장 작은 모델 `yolov8n`(n = nano)입니다.
> - **OK / NG** (No Good): 양품 / 불량. 현장 검사에서 쓰는 판정 용어입니다.
> - **GPU** (Graphics Processing Unit): 그래픽 카드. 학습·추론을 CPU보다 수십 배 빠르게 처리합니다.
> - **CPU** (Central Processing Unit): PC의 중앙 처리 장치. GPU가 없으면 CPU로 돌아가며, 느리지만 동작은 합니다.
> - **CUDA** (Compute Unified Device Architecture): NVIDIA GPU를 계산용으로 쓰게 해주는 플랫폼. PyTorch가 GPU를 쓰려면 CUDA 버전이 맞는 PyTorch가 설치돼 있어야 합니다.
> - **FPS** (Frames Per Second): 초당 처리 프레임 수. 실시간 성능 지표입니다.

---

## 0. 전체 흐름

```
[STEP 0] yolov8_base.py        사전학습 모델로 카메라 데모 (환경 점검)
    │
[STEP 1] 01_capture.py         검사 대상 이미지 촬영        → captures/
    │
[STEP 2] 02_label.py           마우스로 박스 라벨링         → labels/
    │
[STEP 3] 02_split_dataset.py   train/val/test 3:1:1 분할   → dataset/split_날짜시간/
    │
[STEP 4] 03_train.py           YOLOv8n 학습 + test 평가     → run/run_날짜시간/
    │
[STEP 5] 04_infer.py           학습 모델로 실시간 OK/NG 판정 → run/run_날짜시간/infer/
```

각 스크립트는 앞 단계 결과 폴더를 **자동으로 찾아서** 이어받습니다. 경로를 손으로 바꿀 일은 거의 없고, 순서만 지키면 됩니다.

### 권장 진행 시간 (참고)

| 순서 | 내용 | 시간 |
|---|---|---|
| 준비 | 환경 설치·점검 | 20분 |
| STEP 0 | 사전학습 모델 데모 | 10분 |
| STEP 1 | 촬영 | 15분 |
| STEP 2 | 라벨링 | 30분 |
| STEP 3 | 데이터 분할 | 5분 |
| STEP 4 | 학습 (GPU 기준, CPU는 더 김) | 15~30분 |
| STEP 5 | 실시간 추론·결과 해석 | 15분 |

---

## 1. 환경 준비

### 1.1. 준비물

- Windows 10/11 PC (노트북 가능)
- USB 카메라 1대 (노트북 내장 카메라도 가능)
- 검사 대상 물체: 양품(OK) 몇 개 + 불량(NG) 몇 개
  예) 볼트 정상 / 나사산 손상, 라벨 정상 / 오염된 라벨, 펜 뚜껑 닫힘 / 열림 등
- Python **3.12.3**
- 인터넷 연결 (첫 실행 때 패키지·사전학습 모델을 내려받습니다)

### 1.2. 폴더 만들기

압축을 풀어 아래처럼 둡니다. 경로는 예시이며, 다른 곳이어도 됩니다.

```
C:\visionai\seminar\
    01_capture.py
    02_label.py
    02_split_dataset.py
    03_train.py
    04_infer.py
    yolov8_base.py
    README.md
```

> **중요:** 모든 명령은 이 `seminar` 폴더 안에서 실행합니다.
> `01_capture.py`, `02_label.py`는 **현재 명령창 위치 기준**으로 `captures`, `labels` 폴더를 만들기 때문에, 다른 위치에서 실행하면 뒤 단계에서 이미지를 못 찾습니다.

### 1.3. 가상환경 만들기

명령 프롬프트(cmd) 또는 PowerShell을 열고:

```bat
cd C:\visionai\seminar
py -3.12 -m venv .venv
.venv\Scripts\activate
python --version
```

`Python 3.12.x`가 나오면 됩니다. 프롬프트 앞에 `(.venv)`가 붙어 있어야 가상환경이 켜진 상태입니다.

> PowerShell에서 `activate`가 "스크립트를 실행할 수 없습니다" 오류를 내면 한 번만 아래를 실행하고 다시 시도합니다.
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### 1.4. 패키지 설치

**(A) NVIDIA GPU가 있는 PC** — GPU용 PyTorch를 **먼저** 설치하고 나서 ultralytics를 설치합니다. 순서가 바뀌면 CPU용 PyTorch가 깔립니다.

```bat
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics
```

> `cu126`은 CUDA 12.6용이라는 뜻입니다. 그래픽 드라이버가 오래됐으면 `nvidia-smi` 명령으로 표시되는 CUDA Version을 확인하고, [pytorch.org](https://pytorch.org/get-started/locally/)의 설치 안내에서 맞는 명령을 복사해 쓰세요.

**(B) GPU가 없는 PC**

```bat
python -m pip install --upgrade pip
pip install ultralytics
```

**(공통, 선택)** 카메라 이름까지 표시하고 싶으면:

```bat
pip install pygrabber
```

`ultralytics`를 설치하면 `opencv-python`, `numpy`, `torch`가 함께 설치됩니다.

### 1.5. 설치 확인

```bat
python -c "import cv2, torch, ultralytics; print('cv2', cv2.__version__); print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); print('ultralytics', ultralytics.__version__)"
```

출력 예:

```
cv2 4.10.0
torch 2.x.x+cu126 CUDA True
ultralytics 8.x.x
```

- `CUDA True` → GPU로 학습·추론합니다.
- `CUDA False` → CPU로 동작합니다. 오늘 실습은 CPU로도 끝까지 됩니다(학습만 느림).

---

## 2. STEP 0 — 사전학습 모델 데모 (`yolov8_base.py`)

직접 학습하기 전에, COCO 데이터셋(사람·컵·핸드폰 등 80종)으로 이미 학습된 모델로 카메라와 YOLO가 잘 도는지 확인합니다.

```bat
python yolov8_base.py
```

실행하면 순서대로 물어봅니다.

1. **모델 선택** `1` = Object Detection(박스), `2` = Instance Segmentation(물체 윤곽 마스크) → Enter(1)
2. **Backend 선택** `1` = MSMF(Windows 기본), `2` = DSHOW → Enter(1). 영상이 자주 끊기면 다음 실행 때 2를 고릅니다.
3. **카메라 선택** 카메라가 여러 대면 ID를 입력합니다. 1대면 자동 선택.

> **Backend** 는 OpenCV가 카메라에 접근하는 방식(드라이버 경로)입니다.
> - **MSMF** (Microsoft Media Foundation): Windows 최신 방식
> - **DSHOW** (DirectShow): Windows 구형 방식, 호환성이 좋음
>
> Backend마다 카메라 번호 순서가 다를 수 있어서, 이 스크립트는 한 번 고른 Backend를 끝까지 고정합니다.

첫 실행에는 `yolov8n.pt`(또는 `yolov8n-seg.pt`)를 자동으로 내려받습니다.

### 조작키

| 키 | 동작 |
|---|---|
| `Q` / `ESC` | 종료 |
| `C` | 다음 카메라로 전환 |
| `S` | 현재 화면 저장 |
| `+` / `-` | Confidence(신뢰도 임계값) 0.05씩 조정 |

- **Confidence**: 모델이 "이게 맞다"고 확신하는 정도(0~1). 임계값을 올리면 확실한 것만 표시되고, 내리면 애매한 것까지 표시됩니다.
- 오른쪽 패널에 FPS, 검출 개수, 사용 중인 카메라·해상도가 표시됩니다.

**확인 포인트:** 카메라 앞에 사람, 컵, 핸드폰을 비추면 `person`, `cup`, `cell phone` 박스가 뜹니다. 여기까지 되면 환경 준비는 끝입니다.

> ⚠️ **주의 — 여기서는 `S` 키를 누르지 마세요.**
> 이 스크립트도 `captures` 폴더에 저장하는데, 박스와 오른쪽 패널이 그려진 화면이 그대로 저장됩니다. 다음 단계 학습 이미지와 섞이면 학습을 망칩니다.
> 이미 눌렀다면 STEP 1 전에 `captures` 폴더 안의 `cam0_...jpg` 파일을 지우세요.

---

## 3. STEP 1 — 학습 이미지 촬영 (`01_capture.py`)

```bat
python 01_capture.py
```

| 키 | 동작 |
|---|---|
| `Space` | 1장 저장 |
| `a` | 자동 저장 ON/OFF (1초마다 1장) |
| `q` / `ESC` | 종료 |

저장 위치: `captures/img_날짜_시간_번호.jpg`

### 잘 찍는 요령

- **양 목표:** 클래스당 최소 30~50장, 전체 100장 이상이면 데모 결과가 확실히 좋아집니다.
- **OK와 NG를 골고루**: 한쪽만 많으면 적은 쪽을 잘 못 찾습니다.
- **변화를 주면서** 찍습니다: 위치, 각도, 회전, 거리, 한 화면에 여러 개 등. 실제 검사 때 나올 법한 모습을 담는 게 핵심입니다.
- **아무것도 없는 배경 사진**도 10~20장 찍어 두면 오검출(없는데 있다고 하는 것)이 줄어듭니다.
- 조명은 되도록 고정합니다. 현장 비전 검사에서 조명이 성능의 절반입니다.
- 자동 저장(`a`)을 켜 두고 물체를 천천히 돌리면 빠르게 모을 수 있습니다. 단, 거의 똑같은 사진만 수백 장이면 의미가 없습니다.

### 설정 바꾸기 (파일 맨 위 `===== 설정 =====`)

```python
CAM_ID = 0              # 카메라가 여러 대면 1, 2 ...
AUTO_INTERVAL = 1.0     # 자동 저장 간격(초)
FRAME_W, FRAME_H = 1280, 720
```

> 카메라가 1280×720을 지원하지 않으면 가능한 해상도로 자동 조정되고, 실제 해상도가 터미널에 출력됩니다.

---

## 4. STEP 2 — 라벨링 (`02_label.py`)

찍은 이미지마다 물체 위치에 박스를 그리고 클래스를 지정합니다. 결과는 YOLO 형식 텍스트 파일로 저장됩니다.

```bat
python 02_label.py
```

### 조작법

| 입력 | 동작 |
|---|---|
| 마우스 왼쪽 드래그 | 박스 그리기 |
| 마우스 오른쪽 클릭 | 클릭한 위치의 박스 삭제 |
| `0` / `1` | 클래스 선택 (0 = ok, 1 = ng) |
| `d` 또는 `Space` | 저장 후 다음 이미지 |
| `a` | 저장 후 이전 이미지 |
| `z` | 마지막 박스 취소 |
| `c` | 현재 이미지 박스 전체 삭제 |
| `q` / `ESC` | 저장 후 종료 |

- 박스 색: ok = 초록, ng = 빨강
- 화면 위쪽에 `[현재번호/전체] 파일명 class:현재클래스 boxes:박스개수`가 표시됩니다.
- **클래스를 먼저 고르고 박스를 그립니다.** 이미 그린 박스의 클래스는 바뀌지 않으니, 잘못 그렸으면 우클릭으로 지우고 다시 그립니다.
- 중간에 종료해도 됩니다. 다시 실행하면 처음부터 열리지만 이미 그린 박스는 불러오므로 `d`로 빠르게 넘기면 됩니다.

### 라벨링 원칙

- 박스는 물체에 **딱 맞게** 그립니다. 여백이 크면 위치 정확도가 떨어집니다.
- 한 이미지에 물체가 여러 개면 **전부** 박스를 칩니다. 하나라도 빠뜨리면 모델은 "그건 물체가 아니다"라고 배웁니다.
- 기준을 통일합니다. "이 정도 흠집이면 NG"라는 기준이 사람마다 다르면 모델도 헷갈립니다.
- 빈 배경 이미지는 박스 없이 `d`로 넘깁니다. **빈 라벨 파일**이 저장되고, YOLO는 이를 "여기는 아무것도 없음"으로 학습합니다.

### 결과 파일

```
labels/
    classes.txt                       ← ok, ng (클래스 이름 목록)
    img_20260923_101530_0000.txt
    img_20260923_101531_0001.txt
    ...
```

라벨 파일 한 줄 = 박스 하나:

```
1 0.512300 0.433100 0.120500 0.098700
│    │        │        │        └ 박스 높이 (이미지 높이 대비 비율)
│    │        │        └ 박스 너비 (이미지 너비 대비 비율)
│    │        └ 박스 중심 y (0~1)
│    └ 박스 중심 x (0~1)
└ 클래스 번호 (0=ok, 1=ng)
```

좌표를 픽셀이 아닌 0~1 비율로 저장하기 때문에, 학습 때 이미지 크기가 바뀌어도 라벨은 그대로 쓸 수 있습니다.

### 클래스를 바꾸고 싶을 때

```python
CLASSES = ["ok", "ng"]            # 예: ["bolt", "scratch", "dent"]
```

최대 10개(숫자키 0~9)까지 됩니다. 클래스를 바꾸면 `04_infer.py`의 `NG_CLASSES`도 함께 맞춰 줍니다.

---

## 5. STEP 3 — 데이터셋 분할 (`02_split_dataset.py`)

라벨링한 데이터를 학습용 / 검증용 / 시험용으로 3:1:1 나눕니다.

```bat
python 02_split_dataset.py
```

| 구분 | 비율 | 용도 |
|---|---|---|
| **train** | 60% | 모델이 실제로 보고 배우는 데이터 |
| **val** (validation) | 20% | 학습 중간중간 성능을 확인하고 가장 좋은 모델(best.pt)을 고르는 데이터 |
| **test** | 20% | 학습이 끝난 뒤 한 번만 쓰는 최종 시험 데이터. 학습·선택 과정에 전혀 쓰지 않음 |

시험 문제를 미리 보고 공부하면 점수가 의미 없는 것처럼, test는 끝까지 숨겨 둬야 실제 성능에 가까운 숫자가 나옵니다.

### 이 스크립트가 하는 일

1. `captures`의 이미지와 `labels`의 라벨을 이름으로 짝 맞춤 (라벨 없는 이미지는 건너뛰고 목록 출력)
2. **층화 분할**: 개수가 적은 클래스(보통 NG)가 들어간 이미지부터 그룹을 묶어, 각 그룹을 3:1:1로 나눕니다. 불량이 train에만 몰리고 test에는 하나도 없는 상황을 막기 위해서입니다.
3. `SEED = 42`로 고정해서 몇 번을 돌려도 같은 결과로 나뉩니다.
4. `dataset/split_날짜시간/` 에 복사하고 `data.yaml`, `split_report.txt` 생성

### 출력 예

```
클래스: ['ok', 'ng']
그룹       ng:   40장 → train 24 / val 8 / test 8
그룹       ok:   60장 → train 36 / val 12 / test 12
그룹    empty:   15장 → train 9 / val 3 / test 3
train:   69장 | 박스 ok:80, ng:30
val  :   23장 | 박스 ok:25, ng:10
test :   23장 | 박스 ok:26, ng:11

완료: C:\visionai\seminar\dataset\split_20260923_103000
```

### 결과 폴더

```
dataset/split_20260923_103000/
    images/train  images/val  images/test
    labels/train  labels/val  labels/test
    data.yaml            ← 학습 설정 파일 (경로 + 클래스 이름)
    split_report.txt     ← 분할 기록
    classes.txt
```

라벨을 더 추가하고 다시 실행하면 **새 폴더가 하나 더** 생깁니다. 이전 분할은 지워지지 않고, 다음 단계는 가장 최근 폴더를 자동으로 씁니다.

---

## 6. STEP 4 — 학습 (`03_train.py`)

```bat
python 03_train.py
```

처음 실행하면 사전학습 모델 `yolov8n.pt`를 내려받아 `seminar` 폴더에 두고, 이후에는 그 파일을 재사용합니다. 이미 있는 모델에서 출발해 우리 데이터에 맞게 추가 학습하는 방식이라(**전이학습**, Transfer Learning) 적은 데이터로도 쓸 만한 결과가 나옵니다.

### 주요 설정

```python
EPOCHS = 100     # 전체 데이터를 몇 바퀴 반복 학습할지
IMGSZ = 640      # 학습 입력 이미지 크기
BATCH = 16       # 한 번에 묶어 처리하는 이미지 수
```

- 학습 코드에 `patience=30`이 걸려 있어서, 30 에포크 동안 val 성능이 좋아지지 않으면 100을 다 채우지 않고 멈춥니다(조기 종료).
- **CPU로 학습하는 경우** 세미나 시간 안에 끝내려면 `EPOCHS = 30`, `BATCH = 8` 정도로 줄이는 걸 권장합니다.
- **GPU 메모리 부족**(`CUDA out of memory`) 오류가 나면 `BATCH`를 8 → 4로 줄입니다.

### 학습 중 화면 읽는 법

```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100      1.2G      1.021      1.334      1.112         12        640
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all         23         35      0.812      0.743      0.801      0.512
```

- `box_loss`, `cls_loss`, `dfl_loss`: 위치 오차, 분류 오차, 박스 경계 오차. **내려가면** 잘 배우고 있는 겁니다.
- 아래 줄은 val 데이터로 매 에포크 확인한 성능입니다.

### 결과

```
[TEST] Precision=0.850 Recall=0.790 F1=0.819 mAP50=0.832 mAP50-95=0.541
최종 모델: C:\visionai\seminar\run\run_20260923_104500\train\weights\best.pt
```

| 지표 | 뜻 | 현장 표현 |
|---|---|---|
| **Precision** (정밀도) | 모델이 찾았다고 한 것 중 진짜 맞은 비율 | 높을수록 **과검**(멀쩡한 걸 불량이라 함)이 적음 |
| **Recall** (재현율) | 실제 있는 것 중 모델이 찾아낸 비율 | 높을수록 **미검**(불량을 놓침)이 적음 |
| **F1** | Precision과 Recall의 조화평균 | 둘의 균형 점수 |
| **mAP50** (mean Average Precision, IoU 0.5 기준) | 박스가 정답과 50% 이상 겹치면 맞은 것으로 치고 계산한 평균 정밀도 | 종합 검출 성능 |
| **mAP50-95** | IoU 기준을 0.5~0.95로 올려가며 계산한 평균 | 박스 위치까지 얼마나 정확한지 |

> **IoU** (Intersection over Union): 예측 박스와 정답 박스가 겹치는 면적 ÷ 합친 면적. 1이면 완전히 일치합니다.

검사 현장에서는 보통 **Recall(미검)을 더 중요하게** 봅니다. 불량이 고객에게 나가는 게 과검보다 훨씬 비싸기 때문입니다.

### 결과 폴더

```
run/run_20260923_104500/
    data.yaml, split_report.txt   ← 어떤 데이터로 학습했는지 기록
    test_result.txt               ← 최종 test 점수
    train/
        weights/best.pt           ← val 성능이 가장 좋았던 모델 (이걸 씁니다)
        weights/last.pt           ← 마지막 에포크 모델
        results.png               ← 에포크별 loss·성능 그래프
        confusion_matrix.png      ← val 혼동행렬
        val_batch0_pred.jpg       ← val 예측 결과 예시
    test/
        confusion_matrix.png      ← test 혼동행렬
```

**같이 열어볼 것:**
- `train/results.png`: loss가 계속 내려가는지, mAP가 올라가다 멈췄는지 확인
- `test/confusion_matrix.png`: ng를 ok로 본 칸(미검)이 몇 개인지 확인
- `train/val_batch0_pred.jpg`: 실제로 어떻게 박스를 치는지 눈으로 확인

---

## 7. STEP 5 — 실시간 추론 (`04_infer.py`)

```bat
python 04_infer.py
```

가장 최근 학습한 `best.pt`를 자동으로 불러와 카메라 영상에 바로 적용합니다.

| 키 | 동작 |
|---|---|
| `s` | 현재 결과 화면 저장 → `run/run_.../infer/` |
| `q` / `ESC` | 종료 |

- 화면 왼쪽 위: `FPS`, 검출 개수(`objs`)
- `ng` 클래스가 하나라도 잡히면 빨간 글씨로 **NG DETECTED (개수)**, 없으면 초록 **OK**

### 설정

```python
MODEL = None          # None이면 최신 best.pt, 특정 모델을 쓰려면 경로 문자열
CONF = 0.1            # 신뢰도 임계값
NG_CLASSES = {"ng"}   # NG로 판정할 클래스 이름
```

- `CONF = 0.1`은 일부러 낮게 잡은 값입니다. 적은 데이터로 학습한 모델이 확신이 낮더라도 불량을 놓치지 않게 하려는 겁니다. 엉뚱한 곳에 박스가 자주 뜨면(과검) 0.3~0.5로 올려 보면서 과검과 미검의 균형을 직접 확인해 봅니다.
- 이전에 학습한 다른 모델과 비교하려면 `MODEL`에 경로를 넣습니다.
  ```python
  MODEL = r"C:\visionai\seminar\run\run_20260923_104500\train\weights\best.pt"
  ```

### 결과가 기대보다 안 좋을 때

1. **미검이 많다** → NG 이미지를 더 찍고 라벨링 → STEP 3부터 다시
2. **과검이 많다** → 배경·양품 이미지 추가, `CONF` 올리기
3. **학습 때는 잘 되는데 실시간은 안 된다** → 촬영 때와 조명·거리·배경이 다른지 확인
4. 데이터를 추가하면 STEP 1 → 2 → 3 → 4 → 5 순서로 다시 돌리면 되고, 이전 결과는 날짜시간 폴더로 모두 남아 있습니다.

---

## 8. 전체 폴더 구조 (실습 완료 후)

```
seminar/
    01_capture.py  02_label.py  02_split_dataset.py
    03_train.py    04_infer.py  yolov8_base.py
    yolov8n.pt                     ← 사전학습 모델 (자동 다운로드)
    captures/                      ← STEP 1 원본 이미지
    labels/                        ← STEP 2 라벨
    dataset/split_날짜시간/        ← STEP 3 분할 데이터
    run/run_날짜시간/
        train/weights/best.pt      ← STEP 4 학습 모델
        test/                      ← STEP 4 test 평가
        infer/                     ← STEP 5 저장 화면
```

---

## 9. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `카메라를 열 수 없습니다` | Teams, Zoom, 카메라 앱, 브라우저 등 카메라를 쓰는 프로그램을 모두 종료. 카메라가 2대 이상이면 `CAM_ID`를 1, 2로 바꿔 보기 |
| 영상이 몇 초마다 끊김 | USB 허브·연장선 대신 PC 본체에 직접 연결. `yolov8_base.py`는 Backend 2(DSHOW)로 재실행 |
| `이미지가 없습니다` (02_label) | `seminar` 폴더가 아닌 곳에서 실행함. `cd C:\visionai\seminar` 후 다시 실행 |
| `사용 가능한 이미지가 N장뿐입니다` | 라벨링을 끝까지 안 함. `02_label.py`에서 `d`로 넘기거나 `q`로 종료해야 라벨 파일이 저장됨 |
| `분할된 데이터셋(data.yaml)이 없습니다` | STEP 3을 먼저 실행 |
| `모델(best.pt)을 찾을 수 없습니다` | STEP 4 학습이 끝나지 않았거나 중간에 멈춤 |
| `No module named 'ultralytics'` / `'cv2'` | 가상환경이 꺼진 상태. `.venv\Scripts\activate` 후 다시 실행 (프롬프트 앞 `(.venv)` 확인) |
| `CUDA False`인데 GPU가 있음 | CPU용 torch가 설치됨. `pip uninstall -y torch torchvision` 후 1.4 (A)의 torch 설치 명령 다시 실행 |
| `CUDA out of memory` | `03_train.py`의 `BATCH`를 8 또는 4로 |
| 창이 안 뜨고 `cv2.imshow ... not implemented` 오류 | `opencv-python-headless`가 같이 깔림. `pip uninstall -y opencv-python-headless opencv-python` 후 `pip install opencv-python` |
| 학습이 너무 느림 (CPU) | `EPOCHS = 30`, `BATCH = 8`로 줄이기. 이미지 수가 많으면 `IMGSZ = 480`도 가능 |
| 학습 중 `DataLoader worker ... ` 오류 (Windows) | `03_train.py`의 `workers=2`를 `workers=0`으로 |
| 첫 실행에서 다운로드 실패 | 사내망·방화벽 문제. 다른 PC에서 받은 `yolov8n.pt`를 `seminar` 폴더에 복사해 두면 다운로드 없이 진행됨 |

---

## 10. 부록 — 전체 소스 코드

스크립트 파일이 없는 참가자는 아래 코드를 같은 파일명으로 저장해서 쓰면 됩니다. (UTF-8로 저장)


### 10.1. `yolov8_base.py` (STEP 0 — 사전학습 모델 데모)

<details>
<summary>코드 펼치기</summary>

```python
# 파일 경로: C:\visionai\vietnam\seminar\00_modelv8n_V2.py
# 목적: USB Camera 선택 + YOLOv8 Detection / Segmentation 실시간 시각화
# Python 3.12.3
#
# V2 변경 사항
# - Backend(MSMF / DSHOW)를 시작 시 한 번만 선택하고 끝까지 고정
#   (Backend마다 카메라 번호 순서가 달라 다른 카메라로 바뀌던 문제 해결)
# - 재연결 시 같은 카메라 + 같은 Backend로만, 대기 후 여러 번 재시도
# - 프레임 끊김 판단을 횟수 대신 시간(초) 기준으로 변경
# - 정보 표시를 영상 오른쪽 패널로 분리 (검출 라벨 가림 방지)

import os

# 아래 두 설정은 cv2를 import 하기 전에 지정해야 적용됩니다.
# MSMF 하드웨어 변환을 끄면 Windows에서 카메라 Open 속도가 크게 빨라집니다.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

# 연결되지 않은 카메라 번호를 탐색할 때 나오는 OpenCV 경고 로그를 줄입니다.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. 기본 설정
# ============================================================

# 탐색할 카메라 번호 범위입니다. (0 ~ MAX_CAMERA_SCAN-1)
MAX_CAMERA_SCAN = 5

# Detection / Segmentation 모델 파일입니다.
DETECTION_MODEL = "yolov8n.pt"
SEGMENTATION_MODEL = "yolov8n-seg.pt"

# 검출 최소 Confidence 초기값입니다. 실행 중 + / - 키로 조정할 수 있습니다.
CONFIDENCE = 0.25
CONFIDENCE_STEP = 0.05

# NMS의 IoU Threshold입니다.
IOU_THRESHOLD = 0.45

# YOLO 내부 추론 이미지 크기입니다.
IMAGE_SIZE = 640

# 이 시간(초) 동안 프레임이 한 장도 안 들어오면 재연결을 시작합니다.
FRAME_TIMEOUT_SEC = 3.0

# 재연결 최대 시도 횟수와 시도 전 대기 시간(초)입니다.
# MSMF는 release 직후 바로 열면 실패하는 경우가 많아 대기가 필요합니다.
RECONNECT_ATTEMPTS = 5
REOPEN_DELAY_SEC = 1.5

# 오른쪽 정보 패널 폭(픽셀)입니다.
PANEL_WIDTH = 240

# OpenCV 화면 이름입니다.
WINDOW_NAME = "YOLOv8 Vision AI Demo"

# S 키로 저장하는 캡처 이미지 폴더입니다. (스크립트 폴더 아래 captures)
SAVE_DIR = Path(__file__).resolve().parent / "captures"

IS_WINDOWS = sys.platform.startswith("win")


# ============================================================
# 2. 모델 / 연산 장치 / Backend 선택
# ============================================================

def select_model():

    print()
    print("============================================")
    print("YOLOv8 Vision AI Demo")
    print("============================================")
    print("1 : Object Detection")
    print("2 : Instance Segmentation")
    print("============================================")

    while True:
        choice = input("실행할 모델을 선택하세요 [1/2] (Enter=1) : ").strip()

        if choice in ("", "1"):
            return DETECTION_MODEL

        if choice == "2":
            return SEGMENTATION_MODEL

        print("1 또는 2를 입력하세요.")


def select_device():

    # NVIDIA GPU가 있으면 GPU로, 없으면 CPU로 추론합니다.
    try:
        import torch

        if torch.cuda.is_available():
            return 0, torch.cuda.get_device_name(0)

    except Exception:
        pass

    return "cpu", "CPU"


def select_backend():

    # Linux 등에서는 V4L2를 사용합니다.
    if not IS_WINDOWS:
        return "V4L2", cv2.CAP_V4L2

    print()
    print("============================================")
    print("Camera Backend 선택")
    print("============================================")
    print("1 : MSMF  (Windows 기본, 현재 PC에서 동작 확인)")
    print("2 : DSHOW (MSMF에서 자주 끊기면 이쪽으로 시도)")
    print("============================================")
    print("선택한 Backend는 프로그램이 끝날 때까지 바뀌지 않습니다.")

    while True:
        choice = input("Backend를 선택하세요 [1/2] (Enter=1) : ").strip()

        if choice in ("", "1"):
            return "MSMF", cv2.CAP_MSMF

        if choice == "2":
            return "DSHOW", cv2.CAP_DSHOW

        print("1 또는 2를 입력하세요.")


# ============================================================
# 3. 카메라 탐색 / 선택 / 연결
# ============================================================

def get_camera_names(backend_name):

    # pygrabber가 주는 이름 순서는 DSHOW 번호 순서와 같습니다.
    # MSMF는 번호 순서가 다를 수 있으므로 이름을 붙이지 않습니다.
    if backend_name != "DSHOW":
        return []

    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()

    except Exception:
        return []


def read_valid_frame(camera, attempts, interval):

    # 정상 프레임이 들어올 때까지 지정 횟수만큼 읽기를 시도합니다.
    for _ in range(attempts):

        try:
            success, frame = camera.read()
        except cv2.error:
            success, frame = False, None

        if success and frame is not None and frame.size > 0:
            return frame

        time.sleep(interval)

    return None


def scan_cameras(backend):

    backend_name, backend_value = backend

    print()
    print("============================================")
    print(f"카메라 탐색 중... (Backend={backend_name}, ID 0 ~ {MAX_CAMERA_SCAN - 1})")
    print("============================================")

    names = get_camera_names(backend_name)
    cameras = []

    for camera_id in range(MAX_CAMERA_SCAN):

        try:
            camera = cv2.VideoCapture(camera_id, backend_value)
        except cv2.error:
            continue

        if not camera.isOpened():
            camera.release()
            continue

        # 실제 프레임이 들어오는 카메라만 목록에 올립니다.
        frame = read_valid_frame(camera, attempts=20, interval=0.05)
        camera.release()

        # 다음 장치를 열기 전에 드라이버가 정리할 시간을 줍니다.
        time.sleep(0.3)

        if frame is None:
            continue

        height, width = frame.shape[:2]

        if camera_id < len(names):
            name = names[camera_id]
        else:
            name = "USB Camera"

        cameras.append({
            "id": camera_id,
            "name": name,
            "width": width,
            "height": height,
        })

        print(f" -> Camera {camera_id} : {name} / {width} x {height}")

    return cameras


def select_camera(cameras):

    if len(cameras) == 1:
        camera_info = cameras[0]
        print()
        print(f"카메라가 1대만 검출되어 Camera {camera_info['id']}를 사용합니다.")
        return camera_info

    valid_ids = [camera_info["id"] for camera_info in cameras]

    print()
    print("============================================")
    print("사용 가능한 카메라 목록")
    print("============================================")

    for camera_info in cameras:
        print(
            f"{camera_info['id']} : {camera_info['name']} "
            f"({camera_info['width']} x {camera_info['height']})"
        )

    print("============================================")

    while True:
        choice = input(
            f"사용할 Camera ID를 입력하세요 {valid_ids} "
            f"(Enter={valid_ids[0]}) : "
        ).strip()

        if choice == "":
            return cameras[0]

        if choice.isdigit() and int(choice) in valid_ids:
            return cameras[valid_ids.index(int(choice))]

        print("목록에 있는 Camera ID를 입력하세요.")


def open_camera(camera_info, backend):

    # 지정한 카메라를 지정한 Backend로만 엽니다.
    # 다른 Backend로 바꾸면 같은 번호라도 다른 카메라가 열릴 수 있으므로 시도하지 않습니다.
    backend_name, backend_value = backend
    camera_id = camera_info["id"]

    print()
    print(f"[Camera Open] ID={camera_id}, Backend={backend_name}")

    try:
        camera = cv2.VideoCapture(camera_id, backend_value)
    except cv2.error as error:
        print(f" -> Camera Open 오류 : {error}")
        return None

    if not camera.isOpened():
        print(" -> Camera Open 실패")
        camera.release()
        return None

    # 카메라 초기화 시간을 줍니다.
    time.sleep(0.5)

    frame = read_valid_frame(camera, attempts=30, interval=0.1)

    if frame is None:
        print(" -> Frame Read 실패")
        camera.release()
        return None

    height, width = frame.shape[:2]

    # 중요:
    # 여기에서는 camera.set()으로 해상도를 변경하지 않습니다.
    # 현재 PC의 MSMF 스트림은 640x480에서 정상 동작하지만
    # 1280x720 강제 변경 시 스트림이 깨지는 문제가 확인됐기 때문입니다.

    camera_info["width"] = width
    camera_info["height"] = height

    print(f" -> 연결 성공 : {width} x {height}")

    return camera


def reconnect_camera(camera_info, backend):

    # 같은 카메라, 같은 Backend로만 여러 번 재연결을 시도합니다.
    for attempt in range(1, RECONNECT_ATTEMPTS + 1):

        print()
        print(
            f"재연결 시도 {attempt}/{RECONNECT_ATTEMPTS} "
            f"(Camera {camera_info['id']}, {REOPEN_DELAY_SEC:.1f}초 대기 후)"
        )

        time.sleep(REOPEN_DELAY_SEC)

        camera = open_camera(camera_info, backend)

        if camera is not None:
            return camera

    return None


# ============================================================
# 4. 화면 표시 보조 함수
# ============================================================

def draw_text(image, text, position, scale, color, thickness=2):

    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def build_display(plot_image, info_lines):

    # 영상 오른쪽에 정보 패널을 붙여서, 영상 위 검출 라벨을 가리지 않게 합니다.
    height = plot_image.shape[0]

    panel = np.full((height, PANEL_WIDTH, 3), (32, 32, 32), dtype=np.uint8)

    y = 35
    for text, color in info_lines:
        draw_text(panel, text, (15, y), 0.62, color)
        y += 32

    # 단축키 안내는 패널 아래쪽에 표시합니다.
    help_lines = [
        "Q/ESC : Quit",
        "C     : Next Camera",
        "S     : Save Image",
        "+/-   : Confidence",
    ]

    y = height - 15 - (len(help_lines) - 1) * 24
    for text in help_lines:
        draw_text(panel, text, (15, y), 0.5, (200, 200, 200), 1)
        y += 24

    return np.hstack([plot_image, panel])


def save_capture(image, camera_id):

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    file_path = SAVE_DIR / f"cam{camera_id}_{timestamp}.jpg"

    if cv2.imwrite(str(file_path), image):
        print(f"캡처 저장 : {file_path}")
    else:
        print(f"캡처 저장 실패 : {file_path}")


def resize_window(camera_info):

    cv2.resizeWindow(
        WINDOW_NAME,
        camera_info["width"] + PANEL_WIDTH,
        camera_info["height"],
    )


# ============================================================
# 5. 메인 프로그램
# ============================================================

def main():

    model_path = select_model()
    device, device_name = select_device()

    model = YOLO(model_path)

    print()
    print("============================================")
    print("YOLO 모델 로드 완료")
    print(f"Model  : {model_path}")
    print(f"Task   : {model.task}")
    print(f"Device : {device_name}")
    print("============================================")

    # Backend를 한 번 선택하면 끝까지 고정합니다.
    backend = select_backend()
    backend_name = backend[0]

    cameras = scan_cameras(backend)

    if not cameras:
        print()
        print(f"{backend_name} Backend로 사용 가능한 Camera Stream을 찾지 못했습니다.")
        print(
            "카메라 연결 상태를 확인하고, Windows 카메라 / Teams / Zoom / "
            "OBS / Browser 등 카메라를 사용하는 프로그램을 종료하십시오."
        )
        if IS_WINDOWS:
            print("다른 Backend를 선택해서 다시 실행해 보십시오.")
        return

    camera_info = select_camera(cameras)

    # 탐색 때 닫은 장치가 완전히 풀릴 시간을 줍니다.
    time.sleep(0.5)

    camera = open_camera(camera_info, backend)

    if camera is None:
        camera = reconnect_camera(camera_info, backend)

    if camera is None:
        print()
        print(f"Camera {camera_info['id']} 연결에 실패했습니다.")
        return

    print()
    print("============================================")
    print(f"Camera            : {camera_info['id']} ({camera_info['name']})")
    print(f"Camera Backend    : {backend_name}")
    print(f"Camera Resolution : {camera_info['width']} x {camera_info['height']}")
    print(f"YOLO Task         : {model.task}")
    print("--------------------------------------------")
    print("Q / ESC : 종료")
    print("C       : 다음 카메라로 전환")
    print("S       : 현재 화면 캡처 저장")
    print("+ / -   : Confidence 조정")
    print("============================================")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    resize_window(camera_info)

    confidence = CONFIDENCE
    previous_time = time.perf_counter()
    smooth_fps = 0.0

    # 프레임 수신 실패가 시작된 시각입니다. 정상이면 None입니다.
    fail_start = None

    try:
        while True:

            # ------------------------------------------------
            # 프레임 읽기
            # ------------------------------------------------
            try:
                success, frame = camera.read()
            except cv2.error as error:
                print(f"OpenCV Camera Read 오류 : {error}")
                success, frame = False, None

            if not success or frame is None or frame.size == 0:

                now = time.perf_counter()

                if fail_start is None:
                    fail_start = now
                    print("프레임 수신 지연 - 대기 중...")

                if now - fail_start >= FRAME_TIMEOUT_SEC:

                    print(
                        f"{FRAME_TIMEOUT_SEC:.0f}초 동안 프레임이 없어 "
                        f"Camera {camera_info['id']} 재연결을 시작합니다."
                    )

                    camera.release()
                    camera = reconnect_camera(camera_info, backend)

                    if camera is None:
                        print("재연결 실패. 프로그램을 종료합니다.")
                        print(
                            "USB 연결(허브/연장선 제외, PC 본체 직결), "
                            "USB 절전 설정, 다른 Backend 사용을 확인하십시오."
                        )
                        break

                    print("재연결 성공")
                    fail_start = None
                    previous_time = time.perf_counter()
                    smooth_fps = 0.0

                else:
                    time.sleep(0.03)

                # 대기 중에도 창이 응답 없음 상태가 되지 않도록 합니다.
                cv2.waitKey(1)
                continue

            if fail_start is not None:
                print(f"프레임 수신 복구 ({time.perf_counter() - fail_start:.1f}초 지연)")
                fail_start = None

            # ------------------------------------------------
            # YOLO 추론
            # ------------------------------------------------
            results = model.predict(
                source=frame,
                conf=confidence,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False,
            )

            result = results[0]

            plot_image = result.plot(
                conf=True,
                labels=True,
                boxes=True,
                masks=True,
            )

            # ------------------------------------------------
            # FPS 계산 (지수 이동 평균으로 흔들림 완화)
            # ------------------------------------------------
            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0

            if smooth_fps == 0.0:
                smooth_fps = current_fps
            else:
                smooth_fps = smooth_fps * 0.90 + current_fps * 0.10

            # ------------------------------------------------
            # 검출 개수
            # ------------------------------------------------
            object_count = len(result.boxes) if result.boxes is not None else 0
            mask_count = len(result.masks.data) if result.masks is not None else 0

            # ------------------------------------------------
            # 정보 패널 구성
            # ------------------------------------------------
            info_lines = [
                (f"FPS : {smooth_fps:.1f}", (0, 255, 255)),
                (f"Objects : {object_count}", (0, 255, 255)),
            ]

            if model.task == "segment":
                info_lines.append((f"Masks : {mask_count}", (0, 255, 255)))

            info_lines += [
                (f"Task : {model.task}", (255, 255, 0)),
                (f"Conf : {confidence:.2f}", (255, 255, 0)),
                (f"Camera : {camera_info['id']}", (255, 255, 0)),
                (f"Backend : {backend_name}", (255, 255, 0)),
                (f"Size : {camera_info['width']}x{camera_info['height']}", (255, 255, 0)),
            ]

            display_frame = build_display(plot_image, info_lines)

            cv2.imshow(WINDOW_NAME, display_frame)

            # ------------------------------------------------
            # 키 입력 처리
            # ------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("c"), ord("C")):

                if len(cameras) < 2:
                    print("전환할 다른 카메라가 없습니다.")

                else:
                    next_index = (cameras.index(camera_info) + 1) % len(cameras)
                    next_info = cameras[next_index]

                    print(f"Camera {camera_info['id']} -> Camera {next_info['id']} 전환")

                    camera.release()
                    time.sleep(REOPEN_DELAY_SEC)

                    new_camera = open_camera(next_info, backend)

                    if new_camera is None:
                        print("카메라 전환 실패. 이전 카메라로 복귀합니다.")
                        camera = reconnect_camera(camera_info, backend)

                        if camera is None:
                            print("이전 카메라 복귀 실패. 프로그램을 종료합니다.")
                            break

                    else:
                        camera = new_camera
                        camera_info = next_info

                    resize_window(camera_info)
                    previous_time = time.perf_counter()
                    smooth_fps = 0.0
                    fail_start = None

            elif key in (ord("s"), ord("S")):
                save_capture(display_frame, camera_info["id"])

            elif key in (ord("+"), ord("=")):
                confidence = min(0.95, round(confidence + CONFIDENCE_STEP, 2))
                print(f"Confidence : {confidence:.2f}")

            elif key in (ord("-"), ord("_")):
                confidence = max(0.05, round(confidence - CONFIDENCE_STEP, 2))
                print(f"Confidence : {confidence:.2f}")

            # 창의 X 버튼을 눌러 닫은 경우에도 종료합니다.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print()
        print("Ctrl+C 입력으로 종료합니다.")

    finally:
        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

    print()
    print("YOLO Vision AI Demo 종료")


if __name__ == "__main__":
    main()
```

</details>

### 10.2. `01_capture.py` (STEP 1 — 촬영)

<details>
<summary>코드 펼치기</summary>

```python
import time
from pathlib import Path

import cv2

# ===== 설정 =====
CAM_ID = 0                      # 카메라 번호 (USB 카메라 여러 대면 0, 1, 2 ...)
SAVE_DIR = Path("captures")     # 저장 폴더 (한글 경로도 가능)
AUTO_INTERVAL = 1.0             # 자동 저장 간격(초)
FRAME_W, FRAME_H = 1280, 720
WIN_NAME = "capture"


def save_image(path: Path, img) -> bool:
    """한글 경로에서도 저장되도록 imencode + tofile 사용"""
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def open_camera(cam_id: int):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)  # Windows 권장
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)             # 리눅스 / 기본 백엔드
    return cap


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cap = open_camera(CAM_ID)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. CAM_ID를 확인하세요.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"카메라 해상도: {real_w}x{real_h}")
    print("조작: [Space] 1장 저장 / [a] 자동저장 ON/OFF / [q] 또는 [ESC] 종료")

    count = len(list(SAVE_DIR.glob("*.jpg")))
    auto, last = False, 0.0

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임 읽기 실패")
            break

        view = frame.copy()
        cv2.putText(view, f"saved:{count}  auto:{'ON' if auto else 'OFF'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(WIN_NAME, view)
        key = cv2.waitKey(1) & 0xFF

        # 창 X 버튼으로 닫은 경우 종료
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        now = time.time()
        if key == ord(" ") or (auto and now - last >= AUTO_INTERVAL):
            name = SAVE_DIR / f"img_{time.strftime('%Y%m%d_%H%M%S')}_{count:04d}.jpg"
            if save_image(name, frame):
                count += 1
                print("저장:", name)
            else:
                print("저장 실패:", name)
            last = now
        elif key == ord("a"):
            auto = not auto
            last = now
            print("자동 저장:", "ON" if auto else "OFF")
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

</details>

### 10.3. `02_label.py` (STEP 2 — 라벨링)

<details>
<summary>코드 펼치기</summary>

```python
from pathlib import Path

import cv2
import numpy as np

# ===== 설정 =====
IMG_DIR = Path("captures")        # 캡처 이미지 폴더
LABEL_DIR = Path("labels")        # YOLO 라벨(.txt) 저장 폴더
CLASSES = ["ok", "ng"]            # 클래스 이름 (숫자키 0~9로 선택)
MAX_W, MAX_H = 1280, 800          # 화면 표시 최대 크기
WIN_NAME = "labeler"
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255),
          (255, 255, 0), (128, 0, 255), (0, 128, 255), (255, 128, 0), (128, 255, 0)]

state = {"scale": 1.0, "drag": None, "boxes": [], "cur": 0, "w": 1, "h": 1}


def imread_unicode(path: Path):
    """한글 경로에서도 읽히도록 fromfile + imdecode 사용"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_labels(txt: Path, w, h):
    boxes = []
    if not txt.exists():
        return boxes
    for line in txt.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = (float(v) for v in p[1:])
        boxes.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def save_labels(txt: Path, boxes, w, h):
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((min(max(0, x1), w), min(max(0, x2), w)))
        y1, y2 = sorted((min(max(0, y1), h), min(max(0, y2), h)))
        bw, bh = x2 - x1, y2 - y1
        if bw < 3 or bh < 3:
            continue
        lines.append(f"{c} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} "
                     f"{bw / w:.6f} {bh / h:.6f}")
    # 박스가 없어도 빈 파일 저장 → YOLO에서 '배경(정상) 이미지'로 학습됨
    txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def on_mouse(event, x, y, flags, _):
    s = state["scale"]
    ox = min(max(x / s, 0), state["w"])
    oy = min(max(y / s, 0), state["h"])
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drag"] = [ox, oy, ox, oy]
    elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
        state["drag"][2:] = [ox, oy]
    elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
        x1, y1, x2, y2 = state["drag"]
        if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
            state["boxes"].append([state["cur"], min(x1, x2), min(y1, y2),
                                   max(x1, x2), max(y1, y2)])
        state["drag"] = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 우클릭한 위치를 포함하는 박스 중 마지막 것 삭제
        for i in range(len(state["boxes"]) - 1, -1, -1):
            _, x1, y1, x2, y2 = state["boxes"][i]
            if x1 <= ox <= x2 and y1 <= oy <= y2:
                state["boxes"].pop(i)
                break


def draw(img, idx, total, name):
    s = state["scale"]
    view = cv2.resize(img, None, fx=s, fy=s) if s != 1.0 else img.copy()
    for c, x1, y1, x2, y2 in state["boxes"]:
        col = COLORS[c % len(COLORS)]
        cv2.rectangle(view, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), col, 2)
        label = CLASSES[c] if c < len(CLASSES) else str(c)
        cv2.putText(view, label, (int(x1 * s), max(15, int(y1 * s) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    if state["drag"]:
        x1, y1, x2, y2 = state["drag"]
        cv2.rectangle(view, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)),
                      COLORS[state["cur"] % len(COLORS)], 1)
    info = (f"[{idx + 1}/{total}] {name}  class:{state['cur']}({CLASSES[state['cur']]})"
            f"  boxes:{len(state['boxes'])}")
    cv2.putText(view, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(view, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return view


def main():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in IMG_DIR.glob("*") if p.suffix.lower() in exts)
    if not images:
        print(f"이미지가 없습니다: {IMG_DIR.resolve()}")
        return

    (LABEL_DIR / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    print("조작: 드래그=박스 / 우클릭=박스 삭제 / 0~9=클래스 / d,Space=다음 / a=이전")
    print("      z=마지막 박스 취소 / c=전체 삭제 / q,ESC=저장 후 종료")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN_NAME, on_mouse)

    idx = 0
    while 0 <= idx < len(images):
        path = images[idx]
        img = imread_unicode(path)
        if img is None:
            print("읽기 실패, 건너뜀:", path)
            idx += 1
            continue
        h, w = img.shape[:2]
        txt = LABEL_DIR / f"{path.stem}.txt"
        state.update(w=w, h=h, drag=None,
                     scale=min(MAX_W / w, MAX_H / h, 1.0),
                     boxes=load_labels(txt, w, h))

        move = 0
        while move == 0:
            cv2.imshow(WIN_NAME, draw(img, idx, len(images), path.name))
            key = cv2.waitKey(20) & 0xFF
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                move = None
            elif ord("0") <= key <= ord("9") and key - ord("0") < len(CLASSES):
                state["cur"] = key - ord("0")
            elif key in (ord("d"), ord(" ")):
                move = 1
            elif key == ord("a"):
                move = -1
            elif key == ord("z") and state["boxes"]:
                state["boxes"].pop()
            elif key == ord("c"):
                state["boxes"].clear()
            elif key in (ord("q"), 27):
                move = None

        save_labels(txt, state["boxes"], w, h)
        print("저장:", txt)
        if move is None:
            break
        idx = max(0, idx + move)
        if idx >= len(images):
            print("마지막 이미지까지 완료했습니다.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

</details>

### 10.4. `02_split_dataset.py` (STEP 3 — 데이터셋 분할)

<details>
<summary>코드 펼치기</summary>

```python
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

# ===== 설정 =====
BASE = Path(__file__).resolve().parent       # seminar 폴더
IMG_DIR = BASE / "captures"
LABEL_DIR = BASE / "labels"
OUT_ROOT = BASE / "dataset"
RATIO = (3, 1, 1)                            # train : val : test
SEED = 42
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_classes():
    cls_file = LABEL_DIR / "classes.txt"
    if not cls_file.exists():
        raise FileNotFoundError(f"classes.txt가 없습니다: {cls_file}")
    return [n.strip() for n in cls_file.read_text(encoding="utf-8").splitlines() if n.strip()]


def read_label_classes(txt: Path):
    """라벨 파일에 들어있는 클래스 번호 목록"""
    ids = []
    for line in txt.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 5:
            ids.append(int(p[0]))
    return ids


def split_counts(n: int):
    """n개를 RATIO 비율로 나눈 개수 (train, val, test)"""
    total = sum(RATIO)
    n_val = round(n * RATIO[1] / total)
    n_test = round(n * RATIO[2] / total)
    if n >= 3:                     # 3장 이상이면 val/test 최소 1장 보장
        n_val, n_test = max(1, n_val), max(1, n_test)
    n_train = n - n_val - n_test
    if n_train < 1:                # 너무 적으면 train 우선
        n_train, n_val, n_test = n, 0, 0
    return n_train, n_val, n_test


def main():
    names = load_classes()
    print("클래스:", names)

    # 1) 이미지-라벨 짝 맞추기
    images = {p.stem: p for p in IMG_DIR.glob("*") if p.suffix.lower() in IMG_EXTS}
    labels = {p.stem: p for p in LABEL_DIR.glob("*.txt") if p.name != "classes.txt"}

    no_label = sorted(set(images) - set(labels))
    no_image = sorted(set(labels) - set(images))
    if no_label:
        print(f"[건너뜀] 라벨 없는 이미지 {len(no_label)}장: {no_label[:5]}{' ...' if len(no_label) > 5 else ''}")
    if no_image:
        print(f"[건너뜀] 이미지 없는 라벨 {len(no_image)}개: {no_image[:5]}{' ...' if len(no_image) > 5 else ''}")

    pairs = [(images[s], labels[s]) for s in sorted(set(images) & set(labels))]
    if len(pairs) < 3:
        print(f"사용 가능한 이미지가 {len(pairs)}장뿐입니다. 최소 3장 이상 필요합니다.")
        return

    # 2) 층화 그룹: 가장 적게 나온 클래스 기준으로 묶기 (희귀 클래스 우선 배분)
    total_count = Counter()
    info = []
    for img, txt in pairs:
        ids = read_label_classes(txt)
        total_count.update(ids)
        info.append((img, txt, ids))

    groups = defaultdict(list)
    for img, txt, ids in info:
        key = min(set(ids), key=lambda c: total_count[c]) if ids else "empty"
        groups[key].append((img, txt))

    # 3) 그룹별 3:1:1 분할
    random.seed(SEED)
    result = {"train": [], "val": [], "test": []}
    for key in sorted(groups, key=str):
        items = groups[key]
        random.shuffle(items)
        n_tr, n_va, n_te = split_counts(len(items))
        result["train"] += items[:n_tr]
        result["val"] += items[n_tr:n_tr + n_va]
        result["test"] += items[n_tr + n_va:n_tr + n_va + n_te]
        gname = names[key] if isinstance(key, int) and key < len(names) else str(key)
        print(f"그룹 {gname:>8s}: {len(items):4d}장 → train {n_tr} / val {n_va} / test {n_te}")

    # 4) 복사
    out = OUT_ROOT / f"split_{time.strftime('%Y%m%d_%H%M%S')}"
    report = [f"source: {IMG_DIR} / {LABEL_DIR}", f"ratio: {RATIO}, seed: {SEED}", ""]
    for split, items in result.items():
        img_out = out / "images" / split
        lbl_out = out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        box_count = Counter()
        for img, txt in items:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(txt, lbl_out / txt.name)
            box_count.update(read_label_classes(txt))
        detail = ", ".join(f"{names[c] if c < len(names) else c}:{n}"
                           for c, n in sorted(box_count.items()))
        line = f"{split:5s}: {len(items):4d}장 | 박스 {detail or '없음'}"
        print(line)
        report.append(line)

    # 5) data.yaml + 리포트
    yaml_lines = [f"path: {out.as_posix()}", "train: images/train",
                  "val: images/val", "test: images/test", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (out / "split_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    shutil.copy2(LABEL_DIR / "classes.txt", out / "classes.txt")

    print("\n완료:", out)
    print("data.yaml:", out / "data.yaml")


if __name__ == "__main__":
    main()
```

</details>

### 10.5. `03_train.py` (STEP 4 — 학습)

<details>
<summary>코드 펼치기</summary>

```python
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO
#split_*/data.yaml을 모두 찾고, 파일 수정 시각이 가장 최근인 것을 고릅니다.
# ===== 설정 =====
BASE = Path(__file__).resolve().parent       # 실행한 seminar 폴더
DATASET_ROOT = BASE / "dataset"              # split_dataset.py 결과 폴더
DATA_YAML = None      # None이면 최신 dataset/split_*/data.yaml 자동 사용, 직접 지정 시 경로 문자열
RUN_ROOT = BASE / "run"                      # 모든 학습 결과 저장 위치
MODEL_NAME = "yolov8n.pt"                    # 사전학습 nano 모델
SEED = 42
EPOCHS = 100
IMGSZ = 640
BATCH = 16
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_latest_yaml() -> Path | None:
    """dataset/split_*/data.yaml 중 가장 최근 것"""
    cands = sorted(DATASET_ROOT.glob("split_*/data.yaml"),
                   key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS)


def main():
    print("torch", torch.__version__, "| CUDA 사용 가능:", torch.cuda.is_available())
    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        print("GPU:", torch.cuda.get_device_name(0))

    # 1) 분할된 데이터셋 찾기
    yaml = Path(DATA_YAML) if DATA_YAML else find_latest_yaml()
    if yaml is None or not yaml.exists():
        print("분할된 데이터셋(data.yaml)이 없습니다. 먼저 split_dataset.py를 실행하세요.")
        return
    ds_dir = yaml.parent
    counts = {s: count_images(ds_dir / "images" / s) for s in ("train", "val", "test")}
    print("사용 데이터셋:", ds_dir)
    print(f"train {counts['train']}장 / val {counts['val']}장 / test {counts['test']}장")
    if counts["train"] == 0 or counts["val"] == 0:
        print("train 또는 val 이미지가 없습니다. split_dataset.py 결과를 확인하세요.")
        return

    # 2) run 폴더 생성 + 사용한 분할 정보 사본 보관
    run_dir = RUN_ROOT / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("결과 저장 폴더:", run_dir)
    shutil.copy2(yaml, run_dir / "data.yaml")
    report = ds_dir / "split_report.txt"
    if report.exists():
        shutil.copy2(report, run_dir / "split_report.txt")

    # 3) 학습 (사전학습 모델은 seminar 폴더에 받아두고 재사용)
    model_path = BASE / MODEL_NAME
    model = YOLO(str(model_path) if model_path.exists() else MODEL_NAME)
    model.train(data=str(yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                device=device, patience=30, workers=2, seed=SEED,
                project=str(run_dir), name="train", exist_ok=True)

    downloaded = Path(MODEL_NAME)
    if not model_path.exists() and downloaded.exists():
        shutil.move(str(downloaded), model_path)

    best_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        print("best.pt를 찾을 수 없습니다:", best_path)
        return

    lines = [f"dataset: {ds_dir}",
             f"train/val/test: {counts['train']}/{counts['val']}/{counts['test']}",
             f"model: {best_path}"]

    # 4) test 셋으로 최종 성능 평가
    if counts["test"] == 0:
        msg = "[TEST] test 이미지가 없어 평가를 건너뜁니다."
        print(msg)
        lines.append(msg)
    else:
        best = YOLO(str(best_path))
        m = best.val(data=str(yaml), split="test", imgsz=IMGSZ, device=device,
                     project=str(run_dir), name="test", exist_ok=True)
        p, r = m.box.mp, m.box.mr
        f1 = 2 * p * r / (p + r + 1e-9)
        result = (f"[TEST] Precision={p:.3f} Recall={r:.3f} F1={f1:.3f} "
                  f"mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")
        print(result)
        lines.append(result)
        print("혼동행렬:", Path(m.save_dir) / "confusion_matrix.png")

    (run_dir / "test_result.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("최종 모델:", best_path)


if __name__ == "__main__":   # Windows 멀티프로세싱 필수
    main()
```

</details>

### 10.6. `04_infer.py` (STEP 5 — 실시간 추론)

<details>
<summary>코드 펼치기</summary>

```python
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ===== 설정 =====
BASE = Path(__file__).resolve().parent       # 실행한 seminar 폴더
RUN_ROOT = BASE / "run"
MODEL = None          # None이면 최신 run의 best.pt 자동 사용, 직접 지정 시 경로 문자열
CAM_ID = 0
CONF = 0.1            # 신뢰도 임계값 (이 값 미만 검출은 무시)
IMGSZ = 640
FRAME_W, FRAME_H = 1280, 720
NG_CLASSES = {"ng"}   # 불량으로 판정할 클래스 이름
WIN_NAME = "YOLOv8n infer"


def find_latest_best() -> Path | None:
    """run/run_*/train/weights/best.pt 중 가장 최근 것"""
    cands = sorted(RUN_ROOT.glob("run_*/train/weights/best.pt"),
                   key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def save_image(path: Path, img) -> bool:
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def open_camera(cam_id: int):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)  # Windows 권장
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)
    return cap


def main():
    model_path = Path(MODEL) if MODEL else find_latest_best()
    if model_path is None or not model_path.exists():
        print("모델(best.pt)을 찾을 수 없습니다. 먼저 train.py로 학습하세요.")
        return
    print("사용 모델:", model_path)

    device = 0 if torch.cuda.is_available() else "cpu"
    print("장치:", torch.cuda.get_device_name(0) if device == 0 else "CPU")
    model = YOLO(str(model_path))

    # 결과 저장 폴더: 모델이 속한 run 폴더/infer
    run_dir = model_path.parents[2] if model_path.parent.name == "weights" else BASE
    save_dir = run_dir / "infer"
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(CAM_ID)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다. CAM_ID를 확인하세요.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    print("조작: [s] 결과 이미지 저장 / [q] 또는 [ESC] 종료")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    prev, fps = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임 읽기 실패")
            break

        res = model.predict(frame, conf=CONF, imgsz=IMGSZ, device=device,
                            half=(device == 0), verbose=False)[0]
        view = res.plot()                       # 박스+클래스+신뢰도 그리기

        # 결과 값 직접 사용: NG 클래스가 하나라도 있으면 NG 판정
        ng_count = sum(1 for b in res.boxes if model.names[int(b.cls)] in NG_CLASSES)
        if ng_count:
            cv2.putText(view, f"NG DETECTED ({ng_count})", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        else:
            cv2.putText(view, "OK", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3)

        now = time.time()
        inst = 1 / max(now - prev, 1e-6)
        fps = inst if fps == 0 else fps * 0.9 + inst * 0.1   # 이동평균
        prev = now
        cv2.putText(view, f"FPS {fps:.1f}  objs {len(res.boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(WIN_NAME, view)

        k = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            name = save_dir / f"result_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            print("저장:" if save_image(name, view) else "저장 실패:", name)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

</details>
