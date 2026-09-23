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
