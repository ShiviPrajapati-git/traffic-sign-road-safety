"""
verify_gtsrb.py
Read-only integrity check for the extracted GTSRB dataset.

It does NOT modify, move or delete anything. It makes no assumptions about CSV
column names or delimiter: it detects the delimiter (tab or comma) and reports
what it finds. Run from the project root:

    python verify_gtsrb.py
    python verify_gtsrb.py "C:\\some\\other\\GTSRB_folder"     (optional custom path)

A text copy of the report is saved to outputs\\evaluation\\gtsrb_verification.txt
"""
import csv
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import config

EXPECTED_CLASSES = 43
EXPECTED_TRAIN = 39209
EXPECTED_TEST = 12630
IMG_EXT = {".png", ".ppm", ".jpg", ".jpeg", ".bmp"}
MIN_REAL_CSV_BYTES = 100          # a 2-byte CSV is certainly broken

REPORT: list[str] = []
PROBLEMS: list[str] = []


def out(msg: str = "") -> None:
    print(msg)
    REPORT.append(msg)


def problem(msg: str) -> None:
    PROBLEMS.append(msg)
    out(f"  [PROBLEM] {msg}")


def find_root(base: Path):
    """Find the folder that directly contains 'Train'. Handles nested extraction."""
    if (base / "Train").is_dir():
        return base
    candidates = [p.parent for p in base.rglob("Train") if p.is_dir()]
    if not candidates:
        return None
    if len(candidates) > 1:
        out(f"  [WARN] Several folders named 'Train' found: {[str(c) for c in candidates]}")
        out("         Using the shallowest one. Remove duplicate copies later.")
    return sorted(candidates, key=lambda p: len(p.parts))[0]


def list_images(folder: Path, recursive: bool):
    it = folder.rglob("*") if recursive else folder.glob("*")
    return [p for p in it if p.is_file() and p.suffix.lower() in IMG_EXT]


def check_images(files, label: str):
    """Return (empty_files, unreadable_files). Opens every image with PIL."""
    empty, bad = [], []
    for f in tqdm(files, desc=f"Checking {label}", unit="img", leave=False):
        try:
            if f.stat().st_size == 0:
                empty.append(f)
                continue
            with Image.open(f) as im:
                im.verify()
        except Exception:
            bad.append(f)
    return empty, bad


def read_csv_rows(csv_path: Path):
    """Read a CSV, auto-detecting tab or comma as the delimiter."""
    with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        first_line = fh.readline()
        fh.seek(0)
        delim = "\t" if first_line.count("\t") > first_line.count(",") else ","
        rows = [r for r in csv.reader(fh, delimiter=delim) if r]
    return rows, delim


def inspect_csv(csv_path: Path, root: Path, check_class_vs_folder: bool = False):
    """Report size, header, row count, sample rows, path validity and class ids."""
    out(f"\n--- {csv_path.name} ---")
    if not csv_path.exists():
        problem(f"{csv_path.name} does not exist")
        return None
    size = csv_path.stat().st_size
    out(f"  size: {size:,} bytes")
    if size < MIN_REAL_CSV_BYTES:
        problem(f"{csv_path.name} is only {size} bytes: empty or truncated")
        return None
    try:
        rows, delim = read_csv_rows(csv_path)
    except Exception as exc:
        problem(f"could not read {csv_path.name}: {exc}")
        return None
    if len(rows) < 2:
        problem(f"{csv_path.name} has no data rows")
        return None

    header, data = [h.strip() for h in rows[0]], rows[1:]
    out(f"  delimiter detected: {'TAB' if delim == chr(9) else 'COMMA'}")
    out(f"  columns ({len(header)}): {header}")
    out(f"  data rows: {len(data):,}")
    out("  first rows:")
    for r in data[:3]:
        out(f"    {r}")

    # Find a column whose values look like image paths.
    path_col = None
    for ci in range(len(header)):
        sample = [r[ci].strip() for r in data[:50] if ci < len(r) and r[ci].strip()]
        if sample and all(Path(v).suffix.lower() in IMG_EXT for v in sample):
            path_col = ci
            break
    if path_col is None:
        out("  no column looks like image paths (labels may come from folder names instead)")
        return {"rows": len(data), "path_col": None, "header": header}

    # Check that every referenced image exists.
    missing = 0
    for r in data:
        if path_col >= len(r):
            missing += 1
            continue
        rel = r[path_col].strip().replace("\\", "/")
        if not ((root / rel).is_file() or (csv_path.parent / rel).is_file()):
            missing += 1
    out(f"  path column: '{header[path_col]}' | rows whose image file is missing: {missing:,}")
    if missing:
        problem(f"{missing:,} rows in {csv_path.name} point to missing images")

    # Check class ids, if a ClassId column exists.
    class_col = next((i for i, h in enumerate(header) if h.lower() == "classid"), None)
    if class_col is not None:
        bad_ids, mismatched = 0, 0
        for r in data:
            if class_col >= len(r) or path_col >= len(r):
                bad_ids += 1
                continue
            try:
                cid = int(r[class_col])
            except ValueError:
                bad_ids += 1
                continue
            if not 0 <= cid < EXPECTED_CLASSES:
                bad_ids += 1
            if check_class_vs_folder:
                parts = r[path_col].strip().replace("\\", "/").split("/")
                if len(parts) >= 2 and parts[-2] != str(cid):
                    mismatched += 1
        out(f"  ClassId values outside 0..{EXPECTED_CLASSES - 1} or unreadable: {bad_ids:,}")
        if bad_ids:
            problem(f"{bad_ids:,} rows in {csv_path.name} have an invalid ClassId")
        if check_class_vs_folder:
            out(f"  rows where ClassId differs from the class folder in Path: {mismatched:,}")
            if mismatched:
                problem(f"{mismatched:,} rows in {csv_path.name} have ClassId not matching their folder")

    return {"rows": len(data), "path_col": header[path_col], "header": header}


def main() -> int:
    config.setup_logging()
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else config.GTSRB_DIR
    out(f"Checking: {base}")
    if not base.exists():
        out(f"[FAIL] folder does not exist: {base}")
        return 1

    root = find_root(base)
    if root is None:
        out("[FAIL] no 'Train' folder found anywhere under this path.")
        return 1
    out(f"Dataset root: {root}")
    out(f"Contents of root: {sorted(p.name for p in root.iterdir())}")

    # ---- Train ----
    out("\n=== TRAIN IMAGES ===")
    train_dir = root / "Train"
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    names = sorted(d.name for d in class_dirs)
    out(f"  class folders: {len(class_dirs)} (expected {EXPECTED_CLASSES})")
    expected_names = {str(i) for i in range(EXPECTED_CLASSES)}
    if len(class_dirs) != EXPECTED_CLASSES:
        problem(f"found {len(class_dirs)} class folders, expected {EXPECTED_CLASSES}")
    elif {n.lstrip("0") or "0" for n in names} != expected_names:
        out(f"  [WARN] folder names are not 0..42: {names[:10]} ...")

    per_class = {d.name: len(list_images(d, recursive=False)) for d in class_dirs}
    train_files = [f for d in class_dirs for f in list_images(d, recursive=False)]
    out(f"  total train images: {len(train_files):,} (expected {EXPECTED_TRAIN:,})")
    if per_class:
        lo = min(per_class, key=per_class.get)
        hi = max(per_class, key=per_class.get)
        out(f"  smallest class: {lo} ({per_class[lo]} images) | largest: {hi} ({per_class[hi]} images)")
        empty_classes = [k for k, v in per_class.items() if v == 0]
        if empty_classes:
            problem(f"class folders with no images: {empty_classes}")
    if len(train_files) != EXPECTED_TRAIN:
        problem(f"train image count {len(train_files):,} != {EXPECTED_TRAIN:,}")
    empty, bad = check_images(train_files, "Train")
    out(f"  zero-byte images: {len(empty)} | unreadable images: {len(bad)}")
    for f in (empty + bad)[:10]:
        out(f"    {f}")
    if empty or bad:
        problem(f"{len(empty) + len(bad)} train images are empty or unreadable")

    # ---- Test ----
    out("\n=== TEST IMAGES ===")
    test_dir = root / "Test"
    if not test_dir.is_dir():
        problem("'Test' folder not found")
        test_files = []
    else:
        test_files = list_images(test_dir, recursive=True)
        others = [p for p in test_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() not in IMG_EXT]
        out(f"  total test images: {len(test_files):,} (expected {EXPECTED_TEST:,})")
        if others:
            out(f"  non-image files in Test (ignored, not a problem): "
                f"{[(p.name, p.stat().st_size) for p in others[:5]]}")
        if len(test_files) != EXPECTED_TEST:
            problem(f"test image count {len(test_files):,} != {EXPECTED_TEST:,}")
        empty, bad = check_images(test_files, "Test")
        out(f"  zero-byte images: {len(empty)} | unreadable images: {len(bad)}")
        for f in (empty + bad)[:10]:
            out(f"    {f}")
        if empty or bad:
            problem(f"{len(empty) + len(bad)} test images are empty or unreadable")

    # ---- CSVs ----
    out("\n=== CSV FILES ===")
    info = {
        "Train.csv": inspect_csv(root / "Train.csv", root, check_class_vs_folder=True),
        "Test.csv": inspect_csv(root / "Test.csv", root),
        "Meta.csv": inspect_csv(root / "Meta.csv", root),
    }
    if info["Train.csv"] and info["Train.csv"]["rows"] != len(train_files):
        problem(f"Train.csv has {info['Train.csv']['rows']:,} rows but {len(train_files):,} train images exist")
    if info["Test.csv"] and info["Test.csv"]["rows"] != len(test_files):
        problem(f"Test.csv has {info['Test.csv']['rows']:,} rows but {len(test_files):,} test images exist")

    # ---- Meta folder ----
    meta_dir = root / "Meta"
    out("\n=== META FOLDER ===")
    if meta_dir.is_dir():
        out(f"  files: {len(list_images(meta_dir, recursive=True))}")
    else:
        out("  not found (not required for training)")

    # ---- Summary ----
    out("\n=== SUMMARY ===")
    if PROBLEMS:
        out(f"RESULT: {len(PROBLEMS)} PROBLEM(S) FOUND")
        for p in PROBLEMS:
            out(f"  - {p}")
    else:
        out("RESULT: ALL CHECKS PASSED")

    config.ensure_dirs()
    report_path = config.EVALUATION_DIR / "gtsrb_verification.txt"
    report_path.write_text("\n".join(REPORT), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")
    return 1 if PROBLEMS else 0


if __name__ == "__main__":
    sys.exit(main())