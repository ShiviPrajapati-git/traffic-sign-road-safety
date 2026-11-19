"""
src/video_processor.py
End-to-end video pipeline:
  frame -> COCO detection + ByteTrack tracking -> (optional) sign detection + classification
        -> temporal risk analysis -> alert manager -> annotated frame

Writes:
  outputs\\annotated_videos\\<name>_annotated.mp4   annotated video
  outputs\\predictions\\<name>_alerts.csv           alert events (video time, level, message)
  outputs\\predictions\\<name>_risk_timeline.csv    per-frame risk score and level

Run from the project root:
    python src\\video_processor.py --max-frames 150          (quick test, first video found)
    python src\\video_processor.py --source <video.mp4>      (whole video)
    python src\\video_processor.py --no-signs                (skip the sign detector)

Risk values are a rule-based, pixel-based prototype estimate: no real distances, speeds or
collision probabilities.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import logging
import time
from collections import Counter

import cv2
import numpy as np

import config
from src.alert_manager import AlertManager, format_time
from src.risk_analyzer import RiskAnalyzer
from src.sign_detector import SignDetector, draw_signs
from src.tracker import ObjectTracker, draw_tracks

logger = logging.getLogger(__name__)

LEVEL_COLORS = {"LOW": (0, 170, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}  # BGR
ALERT_DISPLAY_SECONDS = 3.0    # how long an alert message stays on screen (video time)
PAIR_LINE_MIN_SCORE = 0.15


def draw_pair_lines(frame, risk, tracks):
    """Draw a line between the pedestrian/vehicle pair that drives the risk (only when risk is not LOW)."""
    if risk["level"] == "LOW":
        return
    by_id = {t["track_id"]: t for t in tracks}
    color = LEVEL_COLORS[risk["level"]]
    for p in risk.get("ped_vehicle_pairs", [])[:1]:
        a, b = by_id.get(p["a_id"]), by_id.get(p["b_id"])
        if a is None or b is None or p["score"] < PAIR_LINE_MIN_SCORE:
            continue
        pa = (int(a["foot"][0]), int(a["foot"][1]))
        pb = (int(b["foot"][0]), int(b["foot"][1]))
        cv2.line(frame, pa, pb, color, 3, cv2.LINE_AA)


def draw_banner(frame, risk, timestamp, alert_text=None):
    """Semi-transparent risk banner at the top, plus a small disclaimer at the bottom."""
    h, w = frame.shape[:2]
    scale = max(0.5, w / 1280.0 * 0.7)
    thick = 2 if w >= 960 else 1
    bar_h = int(70 * scale) + (int(26 * scale) if alert_text else 0)
    color = LEVEL_COLORS[risk["level"]]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (0, 0), (int(14 * scale) + 4, bar_h), color, -1)

    line1 = (f"RISK: {risk['level']}   score {risk['score']:.2f}   "
             f"objects {risk['n_objects']}   t={format_time(timestamp)}")
    cv2.putText(frame, line1, (int(26 * scale) + 4, int(28 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color if risk["level"] != "LOW" else (255, 255, 255),
                thick, cv2.LINE_AA)
    reason = risk["reasons"][0] if risk.get("reasons") else "no elevated risk factors"
    cv2.putText(frame, reason, (int(26 * scale) + 4, int(58 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    if alert_text:
        cv2.putText(frame, "ALERT: " + alert_text, (int(26 * scale) + 4, int(86 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, (0, 215, 255), thick, cv2.LINE_AA)

    cv2.putText(frame, "Prototype rule-based risk estimate (pixel units, uncalibrated)",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale / 0.7 if scale > 0.7 else 0.45,
                (230, 230, 230), 1, cv2.LINE_AA)
    return frame


class VideoProcessor:
    """Runs the full pipeline on a video file. Components can be injected for testing."""

    def __init__(self, tracker=None, sign_detector=None, analyzer=None, alert_manager=None,
                 use_signs: bool = True, sign_every: int = 2):
        self.tracker = tracker or ObjectTracker()
        self.analyzer = analyzer or RiskAnalyzer()
        self.alerts = alert_manager or AlertManager()
        self.sign_every = max(1, int(sign_every))
        self.sign_detector = None
        if use_signs:
            try:
                self.sign_detector = sign_detector or SignDetector()
            except FileNotFoundError as exc:
                logger.warning("Sign detection disabled: %s", exc)
        self._tracker_used = False

    @property
    def signs_enabled(self) -> bool:
        return self.sign_detector is not None

    def process(self, src, out_path=None, max_frames: int = None, progress_cb=None) -> dict:
        """Process one video. progress_cb(done, total) is called every 10 frames (for the dashboard)."""
        src = Path(src)
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open the video: {src}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_target = min(total, max_frames) if max_frames else total

        config.ensure_dirs()
        out_path = Path(out_path) if out_path else config.ANNOTATED_VIDEOS_DIR / f"{src.stem}_annotated.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create the output video: {out_path}")

        if self._tracker_used:
            self.tracker.reset()
        self._tracker_used = True
        self.analyzer.reset()
        self.alerts.reset()

        signs = []
        track_label = {}
        level_frames = Counter()
        sign_counts = Counter()
        timeline = []
        last_alert, last_alert_time = None, -1e9
        peak_score = 0.0
        n = 0
        t_start = time.time()
        try:
            while n < n_target:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = n / fps
                tracks = self.tracker.update(frame)
                if self.signs_enabled and n % self.sign_every == 0:
                    signs = self.sign_detector.detect(frame)
                    sign_counts.update(s["label"] for s in signs if not s["uncertain"])
                risk = self.analyzer.analyze(tracks, signs)
                alert = self.alerts.update(ts, risk, frame_idx=n)
                if alert:
                    last_alert, last_alert_time = alert, ts

                for t in tracks:
                    track_label[t["track_id"]] = t["label"]
                level_frames[risk["level"]] += 1
                peak_score = max(peak_score, risk["score"])
                timeline.append([n, round(ts, 3), risk["level"], risk["score"], risk["raw_score"],
                                 risk["n_objects"]] + [risk["components"][k] for k in
                                 ("ped_vehicle", "vehicle_vehicle", "density", "sign_context")])

                vis = draw_tracks(frame, tracks)
                if signs:
                    vis = draw_signs(vis, signs)
                draw_pair_lines(vis, risk, tracks)
                shown = last_alert["message"] if last_alert and ts - last_alert_time <= ALERT_DISPLAY_SECONDS else None
                draw_banner(vis, risk, ts, shown)
                writer.write(vis)
                n += 1
                if progress_cb and n % 10 == 0:
                    progress_cb(n, n_target)
        finally:
            cap.release()
            writer.release()
        elapsed = max(time.time() - t_start, 1e-6)
        if progress_cb:
            progress_cb(n, n_target)

        alerts_csv = self.alerts.save_csv(config.PREDICTIONS_DIR / f"{src.stem}_alerts.csv")
        timeline_csv = config.PREDICTIONS_DIR / f"{src.stem}_risk_timeline.csv"
        with open(timeline_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_s", "level", "score", "raw_score", "n_objects",
                        "ped_vehicle", "vehicle_vehicle", "density", "sign_context"])
            w.writerows(timeline)

        return {
            "video": str(src), "width": width, "height": height, "fps": fps,
            "frames_processed": n, "duration_s": round(n / fps, 2),
            "processing_fps": round(n / elapsed, 2), "processing_seconds": round(elapsed, 1),
            "signs_enabled": self.signs_enabled, "sign_every": self.sign_every,
            "unique_tracks": len(track_label), "tracks_by_class": dict(Counter(track_label.values())),
            "level_frames": {k: level_frames.get(k, 0) for k in config.RISK_LEVELS},
            "peak_score": round(peak_score, 3),
            "alerts": list(self.alerts.alerts),
            "sign_detections": dict(sign_counts.most_common(8)),
            "annotated_video": str(out_path), "alerts_csv": str(alerts_csv),
            "timeline_csv": str(timeline_csv),
        }


def print_summary(s: dict) -> None:
    n = max(s["frames_processed"], 1)
    print("\n=== VIDEO PROCESSING SUMMARY ===")
    print(f"Video                 : {Path(s['video']).name} ({s['width']}x{s['height']}, {s['fps']:.1f} fps)")
    print(f"Frames processed      : {s['frames_processed']} ({s['duration_s']} s of video)")
    print(f"Processing speed      : {s['processing_fps']} frames/s ({s['processing_seconds']} s total)")
    print(f"Sign detection        : {'on (every ' + str(s['sign_every']) + ' frames)' if s['signs_enabled'] else 'OFF'}")
    print(f"Unique track IDs      : {s['unique_tracks']}   by class: {s['tracks_by_class'] or 'none'}")
    lf = s["level_frames"]
    print("Frames per risk level : " + ", ".join(f"{k} {lf[k]} ({100 * lf[k] / n:.0f}%)" for k in config.RISK_LEVELS))
    print(f"Peak risk score       : {s['peak_score']}")
    print(f"Alerts raised         : {len(s['alerts'])}")
    for a in s["alerts"][:10]:
        print(f"   {a['time_label']}  [{a['type']}]  {a['message']}")
    if len(s["alerts"]) > 10:
        print(f"   ... and {len(s['alerts']) - 10} more (see the CSV)")
    if s["signs_enabled"]:
        print(f"Sign detections (confident, frame-level, not unique signs): {s['sign_detections'] or 'none'}")
    print(f"Annotated video       : {s['annotated_video']}")
    print(f"Alerts CSV            : {s['alerts_csv']}")
    print(f"Risk timeline CSV     : {s['timeline_csv']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full road-safety pipeline on a video")
    parser.add_argument("--source", type=str, default=None, help="video file (default: first in datasets\\test_videos)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sign-every", type=int, default=2, help="run the sign detector every N frames")
    parser.add_argument("--no-signs", action="store_true", help="skip the sign detector")
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

    print(f"Loading models... ({'signs on' if not args.no_signs else 'signs off'})")
    proc = VideoProcessor(use_signs=not args.no_signs, sign_every=args.sign_every)
    if not args.no_signs and not proc.signs_enabled:
        print("[WARN] Sign detection could not start (see the warning above); continuing without it.")
    print(f"Processing {src.name} ...")
    try:
        summary = proc.process(src, max_frames=args.max_frames,
                               progress_cb=lambda d, t: print(f"  {d}/{t} frames") if d % 50 == 0 else None)
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())