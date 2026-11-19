"""
src/sign_detector.py
Two-stage traffic sign pipeline:
  1. YOLOv8 (trained on GTSDB, one class: traffic_sign) finds WHERE signs are.
  2. The GTSRB CNN (TrafficSignClassifier) says WHICH sign each cropped box is.

Test it on images (run from the project root):
    python src\\sign_detector.py                         (20 validation images)
    python src\\sign_detector.py --source <file or folder> --limit 50
Annotated images are saved to outputs\\predictions\\sign_pipeline_check\\
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import time

import cv2
import numpy as np
from ultralytics import YOLO

import config
from src.traffic_sign_classifier import TrafficSignClassifier

logger = logging.getLogger(__name__)

CROP_PAD_RATIO = 0.10      # GTSRB images include a small border around the sign
MIN_CROP_PIXELS = 8        # crops smaller than this (width or height) are not classified
CLS_UNCERTAIN_BELOW = 0.60 # classifier confidence below this is flagged as uncertain


class SignDetector:
    """Detect traffic signs in a BGR frame (OpenCV) and classify each one."""

    def __init__(self, detector_weights=config.SIGN_DETECTOR_WEIGHTS, classifier=None,
                 conf: float = config.SIGN_DET_CONF, imgsz: int = config.SIGN_DET_IMG_SIZE,
                 device: str = None):
        weights = Path(detector_weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"Sign detector weights not found: {weights}\n"
                "Copy your trained weights here, e.g.:\n"
                'Copy-Item "models\\sign_detector\\gtsdb_yolov8n_960-2\\weights\\best.pt" '
                '"models\\traffic_sign_yolo.pt"'
            )
        self.device = device or config.get_device()
        self._yolo_device = 0 if self.device == "cuda" else "cpu"
        self.conf = conf
        self.imgsz = imgsz
        self.detector = YOLO(str(weights))
        self.classifier = classifier or TrafficSignClassifier(device=self.device)
        logger.info("Sign detector ready: weights=%s imgsz=%d conf=%.2f device=%s",
                    weights.name, imgsz, conf, self.device)

    def detect(self, frame_bgr):
        """Return a list of dicts, one per detected sign:
        bbox (x1, y1, x2, y2 ints), det_conf, class_id, label, cls_conf, uncertain."""
        if frame_bgr is None or frame_bgr.ndim != 3:
            return []
        h, w = frame_bgr.shape[:2]
        res = self.detector.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz,
                                    device=self._yolo_device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        xyxy = res.boxes.xyxy.cpu().numpy()
        det_confs = res.boxes.conf.cpu().numpy()

        boxes, crops = [], []
        for (x1, y1, x2, y2), dc in zip(xyxy, det_confs):
            px, py = (x2 - x1) * CROP_PAD_RATIO, (y2 - y1) * CROP_PAD_RATIO
            bx1, by1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
            bx2, by2 = int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2)))
            cx1, cy1 = int(max(0, np.floor(x1 - px))), int(max(0, np.floor(y1 - py)))
            cx2, cy2 = int(min(w, np.ceil(x2 + px))), int(min(h, np.ceil(y2 + py)))
            crop = frame_bgr[cy1:cy2, cx1:cx2]
            if crop.shape[0] < MIN_CROP_PIXELS or crop.shape[1] < MIN_CROP_PIXELS:
                crop = None
            boxes.append(((bx1, by1, bx2, by2), float(dc)))
            crops.append(crop)

        preds = self.classifier.predict_batch(crops)   # None entries stay None
        signs = []
        for (bbox, dc), pred in zip(boxes, preds):
            if pred is None:
                signs.append({"bbox": bbox, "det_conf": dc, "class_id": None,
                              "label": "unclassified", "cls_conf": 0.0, "uncertain": True})
            else:
                signs.append({"bbox": bbox, "det_conf": dc, "class_id": pred["class_id"],
                              "label": pred["label"], "cls_conf": pred["confidence"],
                              "uncertain": pred["confidence"] < CLS_UNCERTAIN_BELOW})
        return signs


def draw_signs(frame_bgr, signs):
    """Return a copy of the frame with boxes and labels drawn (green = confident, orange = uncertain)."""
    out = frame_bgr.copy()
    for s in signs:
        x1, y1, x2, y2 = s["bbox"]
        color = (0, 165, 255) if s["uncertain"] else (0, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{s['label']} {s['cls_conf']:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 4 if y1 - th - base - 4 >= 0 else y2 + th + 4
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw, ty + base), color, -1)
        cv2.putText(out, text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _collect_images(source: Path, limit: int):
    if source.is_file():
        return [source]
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in config.ALLOWED_IMAGE_EXT)
    return files[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the detect -> classify sign pipeline on images")
    parser.add_argument("--source", type=str, default=str(config.GTSDB_YOLO_DIR / "images" / "val"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    config.setup_logging()
    config.ensure_dirs()
    source = Path(args.source)
    if not source.exists():
        print(f"[ERROR] Source not found: {source}")
        return 1
    images = _collect_images(source, args.limit)
    if not images:
        print(f"[ERROR] No images found in {source}")
        return 1

    try:
        detector = SignDetector()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        return 1
    if detector.imgsz != 960:
        print(f"[NOTE] SIGN_DET_IMG_SIZE is {detector.imgsz}, but the detector was trained at 960.")

    out_dir = config.PREDICTIONS_DIR / "sign_pipeline_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_signs = uncertain = with_det = 0
    times = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[WARN] Could not read {path.name}, skipping")
            continue
        t0 = time.time()
        signs = detector.detect(frame)
        times.append((time.time() - t0) * 1000)
        total_signs += len(signs)
        uncertain += sum(s["uncertain"] for s in signs)
        with_det += 1 if signs else 0
        cv2.imwrite(str(out_dir / path.name), draw_signs(frame, signs))
        desc = ", ".join(f"{s['label']} ({s['cls_conf']:.2f})" for s in signs) or "no signs detected"
        print(f"{path.name}: {len(signs)} sign(s) -> {desc}")

    timed = times[1:] if len(times) > 1 else times   # first call includes GPU warm-up
    print("\n=== SIGN PIPELINE SUMMARY ===")
    print(f"Images processed        : {len(times)}")
    print(f"Images with detections  : {with_det}")
    print(f"Total signs detected    : {total_signs}")
    print(f"Uncertain classifications (conf < {CLS_UNCERTAIN_BELOW}): {uncertain}")
    print(f"Mean time per image     : {np.mean(timed):.1f} ms (excluding first image)")
    print(f"Annotated images saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())