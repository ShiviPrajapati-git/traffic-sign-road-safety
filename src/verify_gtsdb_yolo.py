"""
src/verify_gtsdb_yolo.py
Checks the converted GTSDB YOLO dataset before training.
Run from the project root:  python src\verify_gtsdb_yolo.py
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT / "datasets" / "GTSDB_YOLO" / "dataset.yaml"
SPLITS = ("train", "val", "test")


def main() -> int:
    problems = 0

    # 1. YAML
    if not YAML_PATH.is_file():
        print(f"[ERROR] YAML not found: {YAML_PATH}")
        return 1
    cfg = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    print("YAML file :", YAML_PATH)
    print("YAML text :", cfg)
    base = Path(cfg["path"])
    print("Base path exists:", base.is_dir())
    if not base.is_dir():
        print("[ERROR] 'path' in the YAML does not exist.")
        return 1
    names = cfg.get("names", {})
    n_classes = len(names)
    print("Classes   :", names)

    stems_by_split = {}
    for s in SPLITS:
        img_dir = base / cfg[s]
        lbl_dir = base / "labels" / s
        print(f"\n--- {s} ---")
        print("image folder resolves:", img_dir.is_dir(), "| label folder resolves:", lbl_dir.is_dir())
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            problems += 1
            continue

        imgs = {p.stem for p in img_dir.glob("*.png")}
        lbls = {p.stem for p in lbl_dir.glob("*.txt")}
        stems_by_split[s] = imgs
        print(f"images: {len(imgs)} | labels: {len(lbls)}")
        if imgs - lbls:
            print("[WARN] images without label file:", sorted(imgs - lbls)[:5])
            problems += 1
        if lbls - imgs:
            print("[WARN] labels without image:", sorted(lbls - imgs)[:5])
            problems += 1

        empty, boxes, bad_fmt, bad_cls, bad_range = 0, 0, 0, 0, 0
        widths, heights = [], []
        for stem in sorted(lbls):
            text = (lbl_dir / f"{stem}.txt").read_text().strip()
            if not text:
                empty += 1
                continue
            for line in text.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    bad_fmt += 1
                    continue
                try:
                    cid = int(parts[0])
                    xc, yc, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    bad_fmt += 1
                    continue
                boxes += 1
                if not 0 <= cid < n_classes:
                    bad_cls += 1
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    bad_range += 1
                widths.append(w * 1360)   # GTSDB images are 1360x800
                heights.append(h * 800)
        print(f"images with no signs: {empty} | total boxes: {boxes}")
        print(f"bad format lines: {bad_fmt} | bad class ids: {bad_cls} | out-of-range values: {bad_range}")
        if widths:
            widths.sort()
            heights.sort()
            m = len(widths) // 2
            print(f"box width px  : min {widths[0]:.0f} | median {widths[m]:.0f} | max {widths[-1]:.0f}")
            print(f"box height px : min {heights[0]:.0f} | median {heights[m]:.0f} | max {heights[-1]:.0f}")
        problems += bad_fmt + bad_cls + bad_range

    # 2. Leakage between splits (same filename in two splits)
    print("\n--- leakage check ---")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in stems_by_split and b in stems_by_split:
            overlap = stems_by_split[a] & stems_by_split[b]
            print(f"{a} & {b} shared filenames: {len(overlap)}")
            problems += len(overlap)
    for s in SPLITS:
        if s in stems_by_split and stems_by_split[s]:
            ordered = sorted(stems_by_split[s])
            print(f"{s} frame range: {ordered[0]} to {ordered[-1]}")

    print("\n=== RESULT ===")
    print("No problems found." if problems == 0 else f"{problems} problem(s) found. See messages above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())