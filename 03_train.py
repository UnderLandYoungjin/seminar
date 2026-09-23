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
