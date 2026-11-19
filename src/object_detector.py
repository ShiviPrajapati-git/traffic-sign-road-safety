"""
src/object_detector.py
General road-object detection with pretrained YOLOv8 (COCO weights).
Keeps only: pedestrian (COCO 'person'), bicycle, car, motorcycle, bus, truck.
No training is needed here; the pretrained COCO model already knows these classes.

Test it on images (run from the project root):
    python src\\object_detector.py                      (10 GTSDB validation images)
    python src\\object_detector.py --source <file or folder> --limit 20
Annotated images are saved to outputs\\predictions\\object_detector_check\\
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import time
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)

# BGR colors per category (for drawing)
COLORS = {
    "pedestrian": (0, 0, 255),     # red
    "bicycle": (0, 165, 255),      # orange
    "motorcycle": (0, 165, 255),   # orange
    "car": (255, 128, 0),          # blue
    "bus": (255, 0, 255),          # magenta
    "truck": (0, 200, 200),        # yellow-ish
}


class ObjectDetector:
    """Detect road objects in a BGR frame (OpenCV)."""

    def __init__(self, weights=config.OBJECT_DETECTOR_WEIGHTS, conf: float = config.OBJECT_DET_CONF,
                 imgsz: int = 640, device: str = None):
        self.device = device or config.get_device()
        self._yolo_device = 0 if self.device == "cuda" else "cpu"
        self.conf = conf
        self.imgsz = imgsz
        self.class_ids = list(config.COCO_ROAD_CLASSES.keys())
        self.model = YOLO(str(weights))   # auto-downloads yolov8n.pt if it is missing
        logger.info("Object detector ready: weights=%s imgsz=%d conf=%.2f device=%s",
                    weights, imgsz, conf, self.device)

    def detect(self, frame_bgr):
        """Return a list of dicts: bbox (x1, y1, x2, y2 ints), conf, class_id (COCO), label."""
        if frame_bgr is None or frame_bgr.ndim != 3:
            return []
        h, w = frame_bgr.shape[:2]
        res = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz,
                                 classes=self.class_ids, device=self._yolo_device,
                                 verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        objects = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            if k not in config.COCO_ROAD_CLASSES:
                continue
            objects.append({
                "bbox": (int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))),
                "conf": float(c),
                "class_id": int(k),
                "label": config.COCO_ROAD_CLASSES[int(k)],
            })
        return objects


def draw_objects(frame_bgr, objects):
    """Return a copy of the frame with boxes and labels drawn."""
    out = frame_bgr.copy()
    for o in objects:
        x1, y1, x2, y2 = o["bbox"]
        color = COLORS.get(o["label"], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{o['label']} {o['conf']:.2f}"
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
    parser = argparse.ArgumentParser(description="Test the COCO road-object detector on images")
    parser.add_argument("--source", type=str, default=str(config.GTSDB_YOLO_DIR / "images" / "val"))
    parser.add_argument("--limit", type=int, default=10)
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

    detector = ObjectDetector()
    out_dir = config.PREDICTIONS_DIR / "object_detector_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    times = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[WARN] Could not read {path.name}, skipping")
            continue
        t0 = time.time()
        objs = detector.detect(frame)
        times.append((time.time() - t0) * 1000)
        counts.update(o["label"] for o in objs)
        cv2.imwrite(str(out_dir / path.name), draw_objects(frame, objs))
        per_img = Counter(o["label"] for o in objs)
        desc = ", ".join(f"{n} {lbl}" for lbl, n in per_img.items()) or "no road objects"
        print(f"{path.name}: {desc}")

    timed = times[1:] if len(times) > 1 else times   # first call includes GPU warm-up
    print("\n=== OBJECT DETECTOR SUMMARY ===")
    print(f"Images processed     : {len(times)}")
    print(f"Detections by class  : {dict(counts) if counts else 'none'}")
    print(f"Mean time per image  : {np.mean(timed):.1f} ms (excluding first image)")
    print(f"Annotated images saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())