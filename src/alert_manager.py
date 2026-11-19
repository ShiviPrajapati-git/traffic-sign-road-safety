"""
src/alert_manager.py
Turns the per-frame risk level from RiskAnalyzer into a SMALL number of alert events, so a
single risky moment does not create dozens of repeated alerts.

Rules (all times are VIDEO time in seconds = frame_index / fps, not wall-clock time):
  1. LOW never alerts.
  2. An "episode" starts at the first frame at/above the minimum alert level (default MEDIUM).
     The first frame of an episode raises one alert.
  3. If the level later ESCALATES beyond the highest level already alerted in this episode
     (MEDIUM -> HIGH), one more alert is raised. Flapping back and forth does not re-alert.
  4. While HIGH risk is sustained, a reminder alert is raised at most once every
     REMINDER_SECONDS (default 10 s).
  5. An episode ends only after the level has been LOW for the cooldown time
     (config.RISK_CONFIG["alert_cooldown_seconds"], default 3 s). Short dips to LOW
     (shorter than the cooldown) do NOT start a new episode, so they cannot cause repeats.

Alerts describe a rule-based, pixel-based risk estimate. They are not collision predictions.

Synthetic test (no video needed), run from the project root:
    python src\\alert_manager.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import logging

import config

logger = logging.getLogger(__name__)

RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
REMINDER_SECONDS = 10.0      # re-alert interval while HIGH risk continues
CSV_COLUMNS = ["time_s", "time_label", "frame", "type", "level", "score", "message"]


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


class AlertManager:
    """Call update() once per frame, in order. Call reset() for each new video."""

    def __init__(self, cooldown_seconds: float = None, reminder_seconds: float = REMINDER_SECONDS,
                 min_level: str = "MEDIUM"):
        self.cooldown = float(config.RISK_CONFIG["alert_cooldown_seconds"]
                              if cooldown_seconds is None else cooldown_seconds)
        self.reminder = float(reminder_seconds)
        if min_level not in ("MEDIUM", "HIGH"):
            raise ValueError("min_level must be 'MEDIUM' or 'HIGH'")
        self.min_rank = RANK[min_level]
        self.reset()

    def reset(self) -> None:
        self.alerts = []
        self._episode_max = 0        # highest level rank already alerted in the current episode
        self._last_alert_time = None
        self._low_since = None

    def update(self, timestamp: float, risk: dict, frame_idx: int = None):
        """timestamp: video time in seconds. risk: output of RiskAnalyzer.analyze().
        Returns a new alert dict if one is raised on this frame, otherwise None."""
        level = risk["level"]
        rank = RANK[level]

        if rank < self.min_rank:                       # below the alert threshold
            if self._low_since is None:
                self._low_since = timestamp
            if timestamp - self._low_since >= self.cooldown:
                self._episode_max = 0                  # episode over; next rise alerts again
            return None
        self._low_since = None

        kind = None
        if rank > self._episode_max:
            kind = "new" if self._episode_max == 0 else "escalation"
        elif (rank == RANK["HIGH"] and self._last_alert_time is not None
              and timestamp - self._last_alert_time >= self.reminder):
            kind = "reminder"
        if kind is None:
            return None

        self._episode_max = max(self._episode_max, rank)
        self._last_alert_time = timestamp
        reasons = risk.get("reasons") or []
        detail = reasons[0] if reasons else "elevated risk score"
        alert = {
            "time_s": round(float(timestamp), 2),
            "time_label": format_time(timestamp),
            "frame": frame_idx,
            "type": kind,
            "level": level,
            "score": risk.get("score"),
            "message": f"{level} risk: {detail}",
            "reasons": list(reasons),
        }
        self.alerts.append(alert)
        logger.info("ALERT %s [%s] %s", alert["time_label"], kind, alert["message"])
        return alert

    def save_csv(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.alerts)
        return path


# ---------------------------------------------------------------------------
# Synthetic tests (made-up risk levels at 25 fps, no video needed)
# ---------------------------------------------------------------------------
def _simulate(segments, fps=25.0, **kwargs):
    """segments: list of (level, seconds). Returns the list of alerts raised."""
    mgr = AlertManager(**kwargs)
    frame = 0
    for level, seconds in segments:
        for _ in range(int(round(seconds * fps))):
            risk = {"level": level, "score": {"LOW": 0.1, "MEDIUM": 0.5, "HIGH": 0.8}[level],
                    "reasons": ["pedestrian #1 close to car #2"] if level != "LOW" else []}
            mgr.update(frame / fps, risk, frame_idx=frame)
            frame += 1
    return mgr.alerts


def main() -> int:
    config.setup_logging(logging.WARNING)
    print(f"Synthetic tests of alert logic (cooldown {config.RISK_CONFIG['alert_cooldown_seconds']} s, "
          f"HIGH reminder every {REMINDER_SECONDS:.0f} s, 25 fps)\n")
    ok_all = True

    def check(name, alerts, expected_count, expected_types=None):
        nonlocal ok_all
        types = [a["type"] for a in alerts]
        passed = len(alerts) == expected_count and (expected_types is None or types == expected_types)
        ok_all &= passed
        times = [a["time_label"] for a in alerts]
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print(f"        alerts: {len(alerts)} (expected {expected_count}), types {types}, at {times}")

    check("1. all LOW: no alerts", _simulate([("LOW", 20)]), 0)
    check("2. sustained MEDIUM for 20 s: one alert only", _simulate([("MEDIUM", 20)]), 1, ["new"])
    check("3. MEDIUM then HIGH: one alert at each step",
          _simulate([("MEDIUM", 5), ("HIGH", 5)]), 2, ["new", "escalation"])
    check("4. HIGH, 1 s dip to LOW (shorter than cooldown), HIGH again: no repeat",
          _simulate([("HIGH", 2), ("LOW", 1), ("HIGH", 2)]), 1, ["new"])
    check("5. HIGH, 4 s LOW (longer than cooldown), HIGH again: second alert",
          _simulate([("HIGH", 2), ("LOW", 4), ("HIGH", 2)]), 2, ["new", "new"])
    check("6. sustained HIGH for 25 s: first alert + reminders every 10 s",
          _simulate([("HIGH", 25)]), 3, ["new", "reminder", "reminder"])
    check("7. flapping MEDIUM/HIGH/MEDIUM/HIGH: no repeated alerts",
          _simulate([("MEDIUM", 2), ("HIGH", 2), ("MEDIUM", 2), ("HIGH", 2)]), 2, ["new", "escalation"])
    check("8. min_level=HIGH: MEDIUM is ignored",
          _simulate([("MEDIUM", 5), ("HIGH", 3)], min_level="HIGH"), 1, ["new"])

    print("\nALL SYNTHETIC TESTS PASSED" if ok_all else "\nSOME SYNTHETIC TESTS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())