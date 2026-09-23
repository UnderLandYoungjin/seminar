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
