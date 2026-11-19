"""
src/prepare_gtsdb.py
Convert the GTSDB detection dataset (FullIJCNN2013) into YOLO format for training a
single-class "traffic_sign" detector.

What it does
  1. Finds the folder that contains gt.txt and the .ppm images (works with the nested
     FullIJCNN2013\\FullIJCNN2013 layout).
  2. Reads gt.txt (filename;x1;y1;x2;y2;classId) and groups the boxes by image.
  3. Splits the images (not the boxes) into train / val / test.
  4. Converts each .ppm to .png and writes a YOLO label file
     (class x_center y_center width height, all normalized to 0..1; class is always 0).
  5. Writes dataset.yaml, annotations.csv (keeps the original 0..42 class IDs) and a
     picture with boxes drawn so you can visually check the conversion.

Run from the project root:
    python src\\prepare_gtsdb.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import config

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Locating and reading the raw dataset
# ---------------------------------------------------------------------------
def find_source_dir(explicit=None) -> Path:
    """Return the folder that directly contains gt.txt (handles nested extraction)."""
    if explicit:
        p = Path(explicit)
        if not (p / "gt.txt").is_file():
            raise FileNotFoundError(f"gt.txt not found in {p}")
        return p
    if not config.GTSDB_DIR.is_dir():
        raise FileNotFoundError(f"GTSDB folder not found: {config.GTSDB_DIR}")
    hits = sorted(config.GTSDB_DIR.rglob("gt.txt"), key=lambda p: len(p.parts))
    # Prefer a gt.txt that sits next to .ppm images (the full dataset), not a crop subfolder.
    for gt in hits:
        if any(gt.parent.glob("*.ppm")):
            return gt.parent
    raise FileNotFoundError(
        f"No gt.txt with .ppm images beside it was found under {config.GTSDB_DIR}.\n"
        "Check that FullIJCNN2013.zip was extracted completely (use 7-Zip)."
    )


def read_ground_truth(gt_path: Path) -> dict:
    """Parse gt.txt into {filename: [(x1, y1, x2, y2, class_id), ...]}."""
    boxes = defaultdict(list)
    bad_lines = 0
    with open(gt_path, encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            try:
                name = parts[0].strip()
                x1, y1, x2, y2, cid = (int(float(v)) for v in parts[1:6])
            except (ValueError, IndexError):
                bad_lines += 1
                continue
            boxes[name].append((x1, y1, x2, y2, cid))
    if bad_lines:
        print(f"[WARN] {bad_lines} unreadable lines in gt.txt were skipped")
    if not boxes:
        raise ValueError(f"No annotations could be read from {gt_path}")
    return boxes


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_images(names, val_frac: float, test_frac: float, mode: str, seed: int) -> dict:
    """Assign every image to train / val / test.

    block : sorted by filename, first part = train, next = val, last = test.
            Neighbouring frames of a recording are often very similar, so contiguous blocks
            leak less between splits than a random shuffle (only the block edges touch).
    random: shuffled with a fixed seed.
    """
    names = sorted(names)
    n = len(names)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    if mode == "random":
        rng = np.random.RandomState(seed)
        names = [names[i] for i in rng.permutation(n)]
    n_train = n - n_val - n_test
    assignment = {}
    for i, name in enumerate(names):
        assignment[name] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return assignment


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def to_yolo(box, img_w: int, img_h: int):
    """GTSDB box (inclusive pixel corners) -> normalized YOLO (xc, yc, w, h), or None if invalid."""
    x1, y1, x2, y2, _ = box
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    if w <= 1 or h <= 1:
        return None
    return ((x1 + w / 2) / img_w, (y1 + h / 2) / img_h, w / img_w, h / img_h)


def write_dataset_yaml(out_dir: Path) -> Path:
    path = out_dir / "dataset.yaml"
    lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: traffic_sign",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def draw_check_image(samples, out_path: Path) -> None:
    """Save a 2x2 montage of converted images with their boxes drawn (visual sanity check)."""
    tiles = []
    for img_path, label_path in samples[:4]:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        draw = ImageDraw.Draw(im)
        if label_path.is_file():
            for ln in label_path.read_text().splitlines():
                _, xc, yc, bw, bh = (float(v) for v in ln.split())
                draw.rectangle([(xc - bw / 2) * w, (yc - bh / 2) * h,
                                (xc + bw / 2) * w, (yc + bh / 2) * h], outline=(255, 0, 0), width=3)
        tiles.append(im.resize((680, 400)))
    if not tiles:
        return
    canvas = Image.new("RGB", (1360, 800), (30, 30, 30))
    for i, t in enumerate(tiles):
        canvas.paste(t, ((i % 2) * 680, (i // 2) * 400))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert GTSDB to YOLO format")
    parser.add_argument("--src", type=str, default=None, help="folder containing gt.txt and the .ppm images")
    parser.add_argument("--out", type=str, default=str(config.GTSDB_YOLO_DIR))
    parser.add_argument("--split", choices=["block", "random"], default="block")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    config.ensure_dirs()
    try:
        src = find_source_dir(args.src)
        gt = read_ground_truth(src / "gt.txt")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    ppm_files = sorted(p.name for p in src.glob("*.ppm"))
    print(f"Source folder : {src}")
    print(f"Images (.ppm) : {len(ppm_files)}")
    print(f"Annotated imgs: {len(gt)} | boxes in gt.txt: {sum(len(v) for v in gt.values())}")
    if not ppm_files:
        print("[ERROR] No .ppm images found.")
        return 1
    missing = sorted(set(gt) - set(ppm_files))
    if missing:
        print(f"[WARN] {len(missing)} images named in gt.txt are missing on disk (first: {missing[:3]})")
    bad_cls = sorted({b[4] for v in gt.values() for b in v if not 0 <= b[4] < config.NUM_SIGN_CLASSES})
    if bad_cls:
        print(f"[WARN] class IDs outside 0..{config.NUM_SIGN_CLASSES - 1} found in gt.txt: {bad_cls}")

    out_dir = Path(args.out)
    for sub in ("images", "labels"):  # generated folders only; safe to rebuild
        if (out_dir / sub).exists():
            shutil.rmtree(out_dir / sub)
    for kind in ("images", "labels"):
        for s in SPLITS:
            (out_dir / kind / s).mkdir(parents=True, exist_ok=True)

    assignment = split_images(ppm_files, args.val_frac, args.test_frac, args.split, args.seed)

    counts = {s: {"images": 0, "with_signs": 0, "boxes": 0} for s in SPLITS}
    widths, heights, skipped_boxes = [], [], 0
    rows, check_samples = [], []

    for name in tqdm(ppm_files, desc="Converting", unit="img"):
        split = assignment[name]
        stem = Path(name).stem
        try:
            with Image.open(src / name) as im:
                im = im.convert("RGB")
                img_w, img_h = im.size
                img_out = out_dir / "images" / split / f"{stem}.png"
                im.save(img_out)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] could not convert {name}: {exc}")
            continue

        label_lines = []
        for box in gt.get(name, []):
            yolo = to_yolo(box, img_w, img_h)
            if yolo is None:
                skipped_boxes += 1
                continue
            label_lines.append("0 " + " ".join(f"{v:.6f}" for v in yolo))
            widths.append(yolo[2] * img_w)
            heights.append(yolo[3] * img_h)
            rows.append([name, split, box[0], box[1], box[2], box[3], box[4]])
        label_out = out_dir / "labels" / split / f"{stem}.txt"
        label_out.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        counts[split]["images"] += 1
        counts[split]["with_signs"] += int(bool(label_lines))
        counts[split]["boxes"] += len(label_lines)
        if label_lines and len(check_samples) < 4:
            check_samples.append((img_out, label_out))

    with open(out_dir / "annotations.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "split", "x1", "y1", "x2", "y2", "class_id"])
        w.writerows(rows)
    yaml_path = write_dataset_yaml(out_dir)
    check_png = config.EVALUATION_DIR / "gtsdb_conversion_check.png"
    draw_check_image(check_samples, check_png)

    print("\n=== GTSDB -> YOLO CONVERSION SUMMARY ===")
    print(f"split mode      : {args.split} (seed {args.seed})")
    for s in SPLITS:
        c = counts[s]
        print(f"{s:<5}: {c['images']:>4} images | {c['with_signs']:>4} with signs | {c['boxes']:>4} boxes")
    if widths:
        print(f"box width  (px) : min {min(widths):.0f} | median {np.median(widths):.0f} | max {max(widths):.0f}")
        print(f"box height (px) : min {min(heights):.0f} | median {np.median(heights):.0f} | max {max(heights):.0f}")
    if skipped_boxes:
        print(f"[WARN] {skipped_boxes} degenerate boxes were skipped")
    print(f"\ndataset.yaml     : {yaml_path}")
    print(f"annotations.csv  : {out_dir / 'annotations.csv'}")
    print(f"visual check     : {check_png}  (open it: red boxes should sit on the signs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
