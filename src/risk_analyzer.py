"""
src/risk_analyzer.py
Rule-based TEMPORAL risk estimation for road scenes (prototype).

IMPORTANT HONESTY NOTE: everything here works in PIXEL units on a single uncalibrated
camera. It does not measure real distances, speeds, or collision probabilities. The output
is a heuristic risk SCORE (0 to 1) and a level (LOW / MEDIUM / HIGH), not a prediction.

Factors (each scored 0..1 per frame):
  ped_vehicle    pedestrian/bicycle close to a vehicle (counts fully only if the gap is shrinking)
  vehicle_vehicle two vehicles close to each other (counts fully only if the gap is shrinking)
  sign_context   a pedestrian/children/bicycle warning sign with a vulnerable road user near it
  density        many tracked objects: it only AMPLIFIES risk that already exists (by up to 25%);
                 a busy scene with no close or closing objects scores zero

Closeness alone is normal in traffic (queues, crowds); risk comes from objects CLOSING on each other.

Temporal logic:
  persistence    a factor ramps up only after it has held for several frames (a one-frame
                 glitch or ID flicker cannot raise the level)
  smoothing      the final score is a moving average over the last N frames

Alert cooldown (avoid repeated alerts) lives in alert_manager.py, which uses this output.

Synthetic test (no video needed), run from the project root:
    python src\\risk_analyzer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import math
from collections import deque

import config

logger = logging.getLogger(__name__)

# Heuristic weights (NOT calibrated on data).
# A pedestrian-vehicle pair is the main factor and can reach HIGH on its own, but only when it
# is close AND closing. Vehicle proximity alone can reach at most MEDIUM (0.50 < HIGH 0.65).
W_PED_VEHICLE = 1.00
W_VEHICLE_VEHICLE = 0.50
W_SIGN_CONTEXT = 0.10
DENSITY_BOOST = 0.25             # dense scene multiplies existing risk by up to (1 + 0.25)

NOT_APPROACHING_FACTOR = 0.35    # pedestrian-vehicle pair that is NOT closing counts 35%
VV_NOT_CLOSING_FACTOR = 0.25     # vehicle-vehicle pair that is NOT closing counts 25%
APPROACH_LAG_FRAMES = 6          # compare distance now vs this many frames ago
APPROACH_MIN_CLOSING = 0.05      # gap must shrink by >=5% of the mean box diagonal
ACTIVE_MIN_SCORE = 0.15          # a factor "holds" this frame only if its score >= this
DUPLICATE_IOU = 0.80             # boxes overlapping this much are probably one object
RIDER_OVERLAP = 0.40             # pedestrian this much inside a motorcycle box = rider
WARNING_SIGN_IDS = {27, 28, 29}  # GTSRB: Pedestrians, Children crossing, Bicycles crossing
SIGN_MIN_CONF = 0.60             # ignore uncertain sign classifications


def _diag(bbox) -> float:
    return math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])


def _area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _iou(a, b) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


class RiskAnalyzer:
    """Call analyze(tracks, signs) once per frame, in order. Call reset() for each new video."""

    def __init__(self, cfg: dict = None):
        self.cfg = dict(config.RISK_CONFIG if cfg is None else cfg)
        self.reset()

    def reset(self) -> None:
        self._counts = {}
        self._history = deque(maxlen=int(self.cfg["smoothing_window"]))

    # ---- helpers -------------------------------------------------------
    def _ramp(self, name: str, raw: float) -> float:
        """Persistence: returns 0..1 = how long this factor has held (consecutive-ish frames)."""
        count = self._counts.get(name, 0)
        count = count + 1 if raw >= ACTIVE_MIN_SCORE else max(0, count - 1)
        self._counts[name] = count
        return min(1.0, count / float(self.cfg["persistence_frames"]))

    @staticmethod
    def _is_closing(a: dict, b: dict, mean_diag: float) -> bool:
        ta, tb = a.get("trajectory") or [], b.get("trajectory") or []
        n = min(len(ta), len(tb), APPROACH_LAG_FRAMES + 1)
        if n < 3 or mean_diag <= 0:
            return False
        d_now = _dist(ta[-1], tb[-1])
        d_prev = _dist(ta[-n], tb[-n])
        return (d_prev - d_now) / mean_diag >= APPROACH_MIN_CLOSING

    # ---- per-frame factor scores --------------------------------------
    def _ped_vehicle(self, tracks):
        thr = float(self.cfg["ped_vehicle_proximity_ratio"])
        vulnerable = [t for t in tracks if t["label"] in config.VULNERABLE_CLASS_NAMES]
        vehicles = [t for t in tracks if t["label"] in config.VEHICLE_CLASS_NAMES]
        best, pairs = 0.0, []
        for v in vulnerable:
            for c in vehicles:
                if _iou(v["bbox"], c["bbox"]) >= DUPLICATE_IOU:
                    continue
                if c["label"] == "motorcycle" and v["label"] == "pedestrian":
                    pa = _area(v["bbox"])
                    if pa > 0 and _intersection(v["bbox"], c["bbox"]) / pa >= RIDER_OVERLAP:
                        continue   # the person is riding this motorcycle
                mean_diag = (_diag(v["bbox"]) + _diag(c["bbox"])) / 2.0
                if mean_diag <= 0:
                    continue
                ratio = _dist(v["foot"], c["foot"]) / mean_diag
                closeness = max(0.0, 1.0 - ratio / thr)
                if closeness <= 0:
                    continue
                closing = self._is_closing(v, c, mean_diag)
                score = closeness * (1.0 if closing else NOT_APPROACHING_FACTOR)
                pairs.append({"a_id": v["track_id"], "a_label": v["label"],
                              "b_id": c["track_id"], "b_label": c["label"],
                              "gap_ratio": round(ratio, 2), "closing": closing,
                              "score": round(score, 3)})
                best = max(best, score)
        pairs.sort(key=lambda p: p["score"], reverse=True)
        return best, pairs[:3]

    def _vehicle_vehicle(self, tracks):
        thr = float(self.cfg["vehicle_vehicle_proximity_ratio"])
        vehicles = [t for t in tracks if t["label"] in config.VEHICLE_CLASS_NAMES]
        best, pairs = 0.0, []
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                if _iou(a["bbox"], b["bbox"]) >= DUPLICATE_IOU:
                    continue
                mean_diag = (_diag(a["bbox"]) + _diag(b["bbox"])) / 2.0
                if mean_diag <= 0:
                    continue
                ratio = _dist(a["foot"], b["foot"]) / mean_diag
                closeness = max(0.0, 1.0 - ratio / thr)
                if closeness <= 0:
                    continue
                closing = self._is_closing(a, b, mean_diag)
                score = closeness * (1.0 if closing else VV_NOT_CLOSING_FACTOR)
                pairs.append({"a_id": a["track_id"], "a_label": a["label"],
                              "b_id": b["track_id"], "b_label": b["label"],
                              "gap_ratio": round(ratio, 2), "closing": closing,
                              "score": round(score, 3)})
                best = max(best, score)
        pairs.sort(key=lambda p: p["score"], reverse=True)
        return best, pairs[:3]

    def _density(self, tracks) -> float:
        return min(1.0, len(tracks) / float(self.cfg["high_density_count"]))

    def _sign_context(self, tracks, signs) -> float:
        if not signs:
            return 0.0
        ratio_thr = float(self.cfg["sign_approach_ratio"])
        vulnerable = [t for t in tracks if t["label"] in config.VULNERABLE_CLASS_NAMES]
        for s in signs:
            if s.get("class_id") not in WARNING_SIGN_IDS or s.get("cls_conf", 0.0) < SIGN_MIN_CONF:
                continue
            x1, y1, x2, y2 = s["bbox"]
            size = max(x2 - x1, y2 - y1)
            if size <= 0:
                continue
            sc = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            for v in vulnerable:
                if _dist(sc, v["center"]) / size <= ratio_thr:
                    return 1.0
        return 0.0

    # ---- main entry ----------------------------------------------------
    def analyze(self, tracks, signs=None) -> dict:
        """tracks: output of ObjectTracker.update(). signs: output of SignDetector.detect() (optional)."""
        tracks = tracks or []
        pv_raw, pv_pairs = self._ped_vehicle(tracks)
        vv_raw, vv_pairs = self._vehicle_vehicle(tracks)
        dn_raw = self._density(tracks)
        sg_raw = self._sign_context(tracks, signs)

        # Density only "holds" when the scene is at least half as busy as the high-density count
        dn_active = dn_raw if dn_raw >= 0.5 else 0.0
        comp = {
            "ped_vehicle": pv_raw * self._ramp("ped_vehicle", pv_raw),
            "vehicle_vehicle": vv_raw * self._ramp("vehicle_vehicle", vv_raw),
            "density": dn_raw * self._ramp("density", dn_active),
            "sign_context": sg_raw * self._ramp("sign_context", sg_raw),
        }
        base = (W_PED_VEHICLE * comp["ped_vehicle"]
                + W_VEHICLE_VEHICLE * comp["vehicle_vehicle"]
                + W_SIGN_CONTEXT * comp["sign_context"])
        raw_score = min(1.0, base * (1.0 + DENSITY_BOOST * comp["density"]))
        self._history.append(raw_score)
        score = sum(self._history) / len(self._history)

        if score >= self.cfg["high_score"]:
            level = "HIGH"
        elif score >= self.cfg["medium_score"]:
            level = "MEDIUM"
        else:
            level = "LOW"

        reasons = []
        if comp["ped_vehicle"] * W_PED_VEHICLE >= 0.10 and pv_pairs:
            p = pv_pairs[0]
            reasons.append(f"{p['a_label']} #{p['a_id']} close to {p['b_label']} #{p['b_id']}"
                           f" (gap {p['gap_ratio']}x box size{', closing' if p['closing'] else ''})")
        if comp["vehicle_vehicle"] * W_VEHICLE_VEHICLE >= 0.10 and vv_pairs:
            p = vv_pairs[0]
            reasons.append(f"{p['a_label']} #{p['a_id']} close to {p['b_label']} #{p['b_id']}"
                           f" (gap {p['gap_ratio']}x box size{', closing' if p['closing'] else ''})")
        if comp["density"] >= 0.5 and base >= 0.20:
            reasons.append(f"dense scene ({len(tracks)} tracked objects) amplifies risk")
        if comp["sign_context"] > 0:
            reasons.append("warning sign with vulnerable road user nearby")

        return {"level": level, "score": round(score, 3), "raw_score": round(raw_score, 3),
                "components": {k: round(v, 3) for k, v in comp.items()},
                "n_objects": len(tracks),
                "ped_vehicle_pairs": pv_pairs, "vehicle_vehicle_pairs": vv_pairs,
                "reasons": reasons}


# ---------------------------------------------------------------------------
# Synthetic tests (made-up coordinates, no video needed)
# ---------------------------------------------------------------------------
class _Scene:
    """Builds tracker-style track dicts and keeps each track's trajectory, like ObjectTracker does."""

    def __init__(self):
        self.hist = {}

    def make(self, tid, label, cx, foot_y, w, h):
        foot = (float(cx), float(foot_y))
        self.hist.setdefault(tid, []).append(foot)
        bbox = (cx - w / 2.0, foot_y - h, cx + w / 2.0, float(foot_y))
        return {"track_id": tid, "label": label, "bbox": bbox, "conf": 0.9, "foot": foot,
                "center": (float(cx), foot_y - h / 2.0), "trajectory": list(self.hist[tid])}


def _run(frames_fn, n_frames, signs_fn=None):
    analyzer, scene = RiskAnalyzer(), _Scene()
    results = []
    for f in range(n_frames):
        signs = signs_fn(f) if signs_fn else None
        results.append(analyzer.analyze(frames_fn(f, scene), signs))
    return results


def _summary(results):
    levels = [r["level"] for r in results]
    first = {lv: next((i for i, x in enumerate(levels) if x == lv), None) for lv in ("MEDIUM", "HIGH")}
    return levels[-1], max(r["score"] for r in results), first


def main() -> int:
    config.setup_logging()
    cfg = config.RISK_CONFIG
    print("Synthetic tests of the rule-based risk analyzer (made-up pixel coordinates)")
    print(f"thresholds: MEDIUM >= {cfg['medium_score']}, HIGH >= {cfg['high_score']}; "
          f"persistence {cfg['persistence_frames']} frames; smoothing window {cfg['smoothing_window']}\n")
    ok_all = True

    def check(name, results, expect):
        nonlocal ok_all
        final, peak, first = _summary(results)
        passed = expect(final, peak, first)
        ok_all &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print(f"        final level {final}, peak score {peak:.3f}, "
              f"first MEDIUM at frame {first['MEDIUM']}, first HIGH at frame {first['HIGH']}")
        reasons = next((r["reasons"] for r in reversed(results) if r["reasons"]), [])
        if reasons:
            print(f"        last reasons: {reasons}")

    # 1. Empty scene
    check("1. empty scene stays LOW",
          _run(lambda f, s: [], 30),
          lambda fin, pk, fr: fin == "LOW" and pk == 0.0)

    # 2. Pedestrian and car far apart, both standing still
    def far(f, s):
        return [s.make(1, "pedestrian", 100, 500, 40, 100), s.make(2, "car", 900, 500, 200, 120)]
    check("2. pedestrian far from car stays LOW",
          _run(far, 40),
          lambda fin, pk, fr: fin == "LOW" and pk < 0.05)

    # 3. Pedestrian walks toward a parked car and stops right next to it
    def approach(f, s):
        x = min(300 + 16 * f, 840)
        return [s.make(1, "pedestrian", x, 500, 40, 100), s.make(2, "car", 900, 500, 200, 120)]
    check("3. pedestrian approaching car: HIGH only after persistence, then falls back once it stops closing",
          _run(approach, 60),
          lambda fin, pk, fr: fr["HIGH"] is not None and fr["HIGH"] > 20 and fin != "HIGH")

    # 4. One-frame glitch (e.g. ID flicker)
    def glitch(f, s):
        x = 800 if f == 12 else 100
        return [s.make(1, "pedestrian", x, 500, 40, 100), s.make(2, "car", 900, 500, 200, 120)]
    check("4. single-frame glitch does NOT raise the level",
          _run(glitch, 40),
          lambda fin, pk, fr: fin == "LOW" and fr["MEDIUM"] is None)

    # 5. Crowded scene but nobody is close: 12 cars spread far apart (density alone)
    def dense(f, s):
        return [s.make(i, "car", 100 + 300 * (i % 6), 300 + 250 * (i // 6), 100, 60) for i in range(12)]
    check("5. density alone adds nothing (busy but spread-out traffic scores ~0)",
          _run(dense, 40),
          lambda fin, pk, fr: fin == "LOW" and pk < 0.05)

    # 6. Two cars close together: static queue vs one closing on the other
    def queue(f, s):
        return [s.make(1, "car", 400, 500, 200, 120), s.make(2, "car", 520, 500, 200, 120)]
    def closing(f, s):
        x = max(900 - 10 * f, 400)
        return [s.make(1, "car", 280, 500, 200, 120), s.make(2, "car", x, 500, 200, 120)]
    r_queue, r_close = _run(queue, 70), _run(closing, 70)
    pk_q, pk_c = _summary(r_queue)[1], _summary(r_close)[1]
    passed = _summary(r_queue)[0] == "LOW" and pk_c > 3 * pk_q and _summary(r_close)[2]["HIGH"] is None
    ok_all &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] 6. static vehicle queue stays LOW; a vehicle closing on another scores higher but never HIGH")
    print(f"        static queue peak {pk_q:.3f}, closing pair peak {pk_c:.3f}")

    # 7. Person riding a motorcycle must not count as a pedestrian next to a vehicle
    def rider(f, s):
        return [s.make(1, "pedestrian", 500, 450, 40, 100), s.make(2, "motorcycle", 500, 500, 110, 110)]
    check("7. motorcycle rider is ignored as pedestrian-vehicle risk",
          _run(rider, 40),
          lambda fin, pk, fr: fin == "LOW" and pk == 0.0)

    # 8. Pedestrian-warning sign with a pedestrian near it (context factor only)
    sign = {"bbox": (300, 300, 340, 340), "class_id": 27, "label": "Pedestrians", "cls_conf": 0.95,
            "det_conf": 0.9, "uncertain": False}
    def sign_scene(f, s):
        return [s.make(1, "pedestrian", 330, 420, 40, 100)]
    res = _run(sign_scene, 40, signs_fn=lambda f: [sign])
    final, peak, first = _summary(res)
    passed = 0.0 < peak <= W_SIGN_CONTEXT + 1e-6 and first["MEDIUM"] is None
    ok_all &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] 8. warning sign adds a small context score only (peak {peak:.3f})")

    # 9. Crowded street like the busy test clips: tight queue of 12 static cars + pedestrians on the pavement
    def crowd(f, s):
        cars = [s.make(i, "car", 120 + 150 * (i % 6), 330 + 140 * (i // 6), 140, 80) for i in range(12)]
        peds = [s.make(100 + j, "pedestrian", 100 + 180 * j, 640, 30, 80) for j in range(6)]
        return cars + peds
    check("9. crowded street, everything static: no MEDIUM or HIGH",
          _run(crowd, 60),
          lambda fin, pk, fr: fin == "LOW" and fr["MEDIUM"] is None)

    # 10. Pedestrian walking alongside a moving car at a constant gap (not closing)
    def alongside(f, s):
        x = 300 + 6 * f
        return [s.make(1, "pedestrian", x - 100, 500, 40, 100), s.make(2, "car", x, 500, 200, 120)]
    check("10. pedestrian keeping a constant gap beside a car stays LOW (not closing)",
          _run(alongside, 60),
          lambda fin, pk, fr: fin == "LOW" and fr["MEDIUM"] is None)

    print("\nALL SYNTHETIC TESTS PASSED" if ok_all else "\nSOME SYNTHETIC TESTS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())