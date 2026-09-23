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
