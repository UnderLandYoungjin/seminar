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
