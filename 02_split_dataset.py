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
