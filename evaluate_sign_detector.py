"""
evaluate_sign_detector.py
Score the traffic sign DETECTOR (YOLOv8n trained on GTSDB) once on the held-out TEST split
created by src\\prepare_gtsdb.py (datasets\\GTSDB_YOLO\\images\\test).

The detector was trained and its best epoch chosen using the TRAIN and VAL splits only, so the
test split gives an unbiased estimate. Nothing is retrained.

Run from the project root:
    python evaluate_sign_detector.py

Saves to outputs\\evaluation\\:
    sign_detector_test_metrics.json
    sign_detector_test\\            Ultralytics plots (PR curve, confusion matrix, sample predictions)

Metric notes (Ultralytics, single class "traffic_sign"):
  precision / recall : at the confidence threshold that maximizes F1 on this split
  mAP50              : average precision at IoU 0.50
  mAP50-95           : average precision over IoU 0.50 to 0.95 (stricter; measures box tightness)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
from datetime import datetime

import config


def count_split(split: str):
    """Number of test images and labelled boxes in the converted dataset."""
    img_dir = config.GTSDB_YOLO_DIR / "images" / split
    lab_dir = config.GTSDB_YOLO_DIR / "labels" / split
    n_img = len(list(img_dir.glob("*.png"))) if img_dir.is_dir() else 0
    n_box = n_img_with = 0
    if lab_dir.is_dir():
        for f in lab_dir.glob("*.txt"):
            k = sum(1 for ln in f.read_text().splitlines() if ln.strip())
            n_box += k
            n_img_with += int(k > 0)
    return n_img, n_img_with, n_box


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the sign detector on the GTSDB test split")
    parser.add_argument("--weights", type=str, default=str(config.SIGN_DETECTOR_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=config.SIGN_DET_IMG_SIZE)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    args = parser.parse_args()

    weights = Path(args.weights)
    data_yaml = config.GTSDB_YOLO_DIR / "dataset.yaml"
    if not weights.is_file():
        print(f"[ERROR] Weights not found: {weights}")
        return 1
    if not data_yaml.is_file():
        print(f"[ERROR] {data_yaml} not found. Run: python src\\prepare_gtsdb.py")
        return 1
    n_img, n_img_with, n_box = count_split(args.split)
    if n_img == 0:
        print(f"[ERROR] No images in datasets\\GTSDB_YOLO\\images\\{args.split}")
        return 1

    from ultralytics import YOLO

    config.ensure_dirs()
    print(f"Weights : {weights}")
    print(f"Split   : {args.split} - {n_img} images ({n_img_with} contain signs), {n_box} labelled boxes")
    print(f"Image size {args.imgsz}\n")

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, batch=args.batch,
                        workers=0, plots=True, project=str(config.EVALUATION_DIR),
                        name=f"sign_detector_{args.split}", exist_ok=True)

    result = {
        "weights": str(weights), "split": args.split, "imgsz": args.imgsz,
        "n_images": n_img, "n_images_with_signs": n_img_with, "n_boxes": n_box,
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = config.EVALUATION_DIR / f"sign_detector_{args.split}_metrics.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== SIGN DETECTOR RESULTS ===")
    print(f"Split       : {args.split} ({n_img} images, {n_box} boxes)")
    print(f"Precision   : {result['precision']:.4f}")
    print(f"Recall      : {result['recall']:.4f}")
    print(f"mAP50       : {result['mAP50']:.4f}")
    print(f"mAP50-95    : {result['mAP50_95']:.4f}")
    print(f"\nSaved: {out}")
    print(f"Plots: {config.EVALUATION_DIR / ('sign_detector_' + args.split)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
