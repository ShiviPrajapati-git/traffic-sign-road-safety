"""
src/tracker.py
Multi-object tracking for road objects using Ultralytics YOLOv8 (COCO) + ByteTrack.
Each object gets a track ID that stays the same across frames, plus a short
trajectory (bottom-center "foot" point per frame) that the risk analyzer will use.

Test it on a video (run from the project root):
    python src\\tracker.py --max-frames 150          (quick test on the first video found)
    python src\\tracker.py --source <video.mp4>      (whole video)
Annotated video is saved to outputs\\annotated_videos\\<name>_tracked.mp4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import time
from collections import Counter, defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)

STALE_TRACK_FRAMES = 60   # forget a trajectory if the track has not been seen for this many frames


class ObjectTracker:
    """Detect + track road objects frame by frame. Call update(frame) once per frame, in order."""

    def __init__(self, weights=config.OBJECT_DETECTOR_WEIGHTS, conf: float = config.OBJECT_DET_CONF,
                 imgsz: int = 640, tracker_cfg: str = config.TRACKER_CONFIG, device: str = None):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.tracker_cfg = tracker_cfg
        self.device = device or config.get_device()
        self._yolo_device = 0 if self.device == "cuda" else "cpu"
        self.class_ids = list(config.COCO_ROAD_CLASSES.keys())
        self.reset()

    def reset(self) -> None:
        """Start fresh (new tracker state, empty trajectories). Call before each new video."""
        self.model = YOLO(str(self.weights))
        self.trajectories = defaultdict(lambda: deque(maxlen=config.TRAJECTORY_LENGTH))
        self.last_seen = {}
        self.frame_idx = -1

    def update(self, frame_bgr):
        """Return a list of dicts: track_id, bbox (x1, y1, x2, y2), conf, class_id, label,
        center (cx, cy), foot (fx, fy), trajectory (list of foot points, oldest first)."""
        self.frame_idx += 1
        if frame_bgr is None or frame_bgr.ndim != 3:
            return []
        h, w = frame_bgr.shape[:2]
        res = self.model.track(frame_bgr, persist=True, tracker=self.tracker_cfg, conf=self.conf,
                               imgsz=self.imgsz, classes=self.class_ids, device=self._yolo_device,
                               verbose=False)[0]
        tracks = []
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, c, k in zip(xyxy, ids, confs, clss):
                if k not in config.COCO_ROAD_CLASSES:
                    continue
                x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
                x2, y2 = min(float(w), float(x2)), min(float(h), float(y2))
                foot = ((x1 + x2) / 2.0, y2)
                self.trajectories[int(tid)].append(foot)
                self.last_seen[int(tid)] = self.frame_idx
                tracks.append({
                    "track_id": int(tid),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": float(c),
                    "class_id": int(k),
                    "label": config.COCO_ROAD_CLASSES[int(k)],
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "foot": foot,
                    "trajectory": list(self.trajectories[int(tid)]),
                })
        self._prune()
        return tracks

    def _prune(self) -> None:
        """Drop trajectories of tracks not seen recently (keeps memory bounded on long videos)."""
        stale = [t for t, last in self.last_seen.items() if self.frame_idx - last > STALE_TRACK_FRAMES]
        for t in stale:
            self.trajectories.pop(t, None)
            self.last_seen.pop(t, None)


def _id_color(track_id: int):
    """A stable bright BGR color per track ID."""
    rng = np.random.RandomState(track_id * 7919 % (2 ** 31))
    return tuple(int(v) for v in rng.randint(60, 255, size=3))


def draw_tracks(frame_bgr, tracks):
    """Return a copy of the frame with boxes, ID labels and trajectory trails drawn."""
    out = frame_bgr.copy()
    for t in tracks:
        color = _id_color(t["track_id"])
        x1, y1, x2, y2 = t["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        pts = np.array(t["trajectory"], dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, color, 2, cv2.LINE_AA)
        text = f"ID {t['track_id']} {t['label']} {t['conf']:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 4 if y1 - th - base - 4 >= 0 else y2 + th + 4
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw, ty + base), color, -1)
        cv2.putText(out, text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Test ByteTrack tracking on a video")
    parser.add_argument("--source", type=str, default=None,
                        help="video file (default: first .mp4 in datasets\\test_videos)")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames (quick test)")
    args = parser.parse_args()

    config.setup_logging()
    config.ensure_dirs()

    if args.source:
        src = Path(args.source)
    else:
        vids = sorted(p for p in config.TEST_VIDEOS_DIR.iterdir()
                      if p.suffix.lower() in config.ALLOWED_VIDEO_EXT) if config.TEST_VIDEOS_DIR.exists() else []
        if not vids:
            print(f"[ERROR] No videos found in {config.TEST_VIDEOS_DIR}")
            return 1
        src = vids[0]
    if not src.is_file():
        print(f"[ERROR] Video not found: {src}")
        return 1

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[ERROR] OpenCV could not open: {src}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_target = min(total, args.max_frames) if args.max_frames else total

    out_path = config.ANNOTATED_VIDEOS_DIR / f"{src.stem}_tracked.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"[ERROR] Could not create output video: {out_path}")
        cap.release()
        return 1

    tracker = ObjectTracker()
    print(f"Video: {src.name} ({width}x{height}, {fps:.1f} fps, {total} frames); processing {n_target} frames")

    track_label = {}                   # track_id -> class label
    track_frames = Counter()           # track_id -> number of frames seen
    per_frame_counts = []
    t_start = time.time()
    n = 0
    while n < n_target:
        ok, frame = cap.read()
        if not ok:
            break
        tracks = tracker.update(frame)
        for t in tracks:
            track_label[t["track_id"]] = t["label"]
            track_frames[t["track_id"]] += 1
        per_frame_counts.append(len(tracks))
        writer.write(draw_tracks(frame, tracks))
        n += 1
        if n % 50 == 0:
            print(f"  processed {n}/{n_target} frames")
    elapsed = time.time() - t_start
    cap.release()
    writer.release()

    print("\n=== TRACKER SUMMARY ===")
    print(f"Frames processed       : {n}")
    print(f"Processing speed       : {n / elapsed:.1f} frames/s ({elapsed:.1f} s total)")
    print(f"Unique track IDs       : {len(track_label)}")
    print(f"Tracks by class        : {dict(Counter(track_label.values())) or 'none'}")
    if track_frames:
        longest_id, longest_n = track_frames.most_common(1)[0]
        print(f"Longest track          : ID {longest_id} ({track_label[longest_id]}), {longest_n} frames")
        short = sum(1 for v in track_frames.values() if v < 5)
        print(f"Very short tracks (<5 frames): {short}  (often false detections or ID fragments)")
    print(f"Mean objects per frame : {np.mean(per_frame_counts):.1f}  (max {max(per_frame_counts)})")
    print(f"Annotated video saved  : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())