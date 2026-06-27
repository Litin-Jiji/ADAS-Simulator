"""
collision_risk.py
Collision Risk Assessment — Milestone 5

For each tracked + depth-estimated object:
  1. Track distance over time → compute closing speed
  2. TTC = distance / closing_speed
  3. Assign risk tier: LOW / MEDIUM / HIGH / CRITICAL
  4. Return warnings for HUD + event logging
"""

from dataclasses import dataclass, field
from collections import deque
import time


# ── Risk tiers ─────────────────────────────────────────────────────────────
RISK_LOW      = "LOW"
RISK_MEDIUM   = "MEDIUM"
RISK_HIGH     = "HIGH"
RISK_CRITICAL = "CRITICAL"

# TTC thresholds (seconds)
TTC_CRITICAL = 1.5
TTC_HIGH     = 2.5
TTC_MEDIUM   = 4.0

# Distance thresholds (metres) — risk even if closing speed is low
DIST_CRITICAL = 3.0
DIST_HIGH     = 6.0
DIST_MEDIUM   = 12.0


@dataclass
class RiskResult:
    track_id:      int
    class_name:    str
    distance_m:    float
    closing_speed: float      # m/s, positive = approaching
    ttc:           float      # seconds, -1 = not applicable
    risk:          str        # LOW / MEDIUM / HIGH / CRITICAL
    bbox:          list[int]


@dataclass
class CollisionWarning:
    """Highest risk object in the scene."""
    active:        bool
    risk:          str
    ttc:           float
    distance_m:    float
    track_id:      int
    class_name:    str
    message:       str


class CollisionRiskAssessor:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        # Per-track distance history: track_id → deque of (timestamp, distance)
        self._dist_history: dict[int, deque] = {}
        self._history_len  = 10   # frames to keep for speed estimation

    # ── Speed estimation ───────────────────────────────────────────────────

    def _closing_speed(self, track_id: int, distance_m: float) -> float:
        """
        Estimate closing speed (m/s) using linear regression over
        recent distance history. Positive = object approaching.
        """
        now = time.perf_counter()

        if track_id not in self._dist_history:
            self._dist_history[track_id] = deque(maxlen=self._history_len)

        self._dist_history[track_id].append((now, distance_m))
        history = self._dist_history[track_id]

        if len(history) < 3:
            return 0.0

        times  = [h[0] - history[0][0] for h in history]
        dists  = [h[1] for h in history]

        # Linear fit: slope = rate of distance change
        # Negative slope = distance decreasing = approaching
        try:
            import numpy as np
            coeffs = np.polyfit(times, dists, 1)
            slope  = coeffs[0]          # m/s
            return -slope               # positive = closing
        except Exception:
            return 0.0

    # ── Risk classification ────────────────────────────────────────────────

    def _classify(self, distance_m: float, closing_speed: float) -> tuple[str, float]:
        """Returns (risk_tier, ttc_seconds)."""

        # TTC only meaningful if object is approaching
        if closing_speed > 0.1:
            ttc = distance_m / closing_speed
        else:
            ttc = 99.0   # not approaching

        # Distance-based floor (object too close regardless of speed)
        if distance_m <= DIST_CRITICAL:
            return RISK_CRITICAL, ttc
        if distance_m <= DIST_HIGH:
            return RISK_HIGH, ttc

        # TTC-based classification
        if ttc <= TTC_CRITICAL:
            return RISK_CRITICAL, ttc
        if ttc <= TTC_HIGH:
            return RISK_HIGH, ttc
        if ttc <= TTC_MEDIUM or distance_m <= DIST_MEDIUM:
            return RISK_MEDIUM, ttc

        return RISK_LOW, ttc

    # ── Main assess ───────────────────────────────────────────────────────

    def assess(self, tracked: list[dict]) -> tuple[list[RiskResult], CollisionWarning]:
        """
        Assess collision risk for all tracked objects.

        Returns:
            results  : per-object RiskResult list
            warning  : highest risk CollisionWarning for HUD
        """
        results = []

        for obj in tracked:
            dist = obj.get("distance_m", -1)
            if dist < 0:
                continue

            tid   = obj["track_id"]
            cls   = obj["class_name"]
            bbox  = obj["bbox"]

            closing = self._closing_speed(tid, dist)
            risk, ttc = self._classify(dist, closing)

            results.append(RiskResult(
                track_id=tid,
                class_name=cls,
                distance_m=dist,
                closing_speed=round(closing, 2),
                ttc=round(ttc, 1),
                risk=risk,
                bbox=bbox,
            ))

        # Clean up stale tracks
        active_ids = {obj["track_id"] for obj in tracked}
        for tid in list(self._dist_history.keys()):
            if tid not in active_ids:
                del self._dist_history[tid]

        # Find highest risk object
        warning = CollisionWarning(
            active=False, risk=RISK_LOW, ttc=99.0,
            distance_m=99.0, track_id=-1, class_name="", message=""
        )

        priority = [RISK_CRITICAL, RISK_HIGH, RISK_MEDIUM, RISK_LOW]
        for tier in priority:
            candidates = [r for r in results if r.risk == tier]
            if candidates:
                # Closest object at this risk tier
                worst = min(candidates, key=lambda r: r.distance_m)
                ttc_str = f"TTC {worst.ttc:.1f}s" if worst.ttc < 99 else "Stationary"
                warning = CollisionWarning(
                    active     = tier != RISK_LOW,
                    risk       = tier,
                    ttc        = worst.ttc,
                    distance_m = worst.distance_m,
                    track_id   = worst.track_id,
                    class_name = worst.class_name,
                    message    = f"{tier} RISK  |  {worst.class_name} #{worst.track_id}"
                                 f"  |  {worst.distance_m:.1f}m  |  {ttc_str}",
                )
                break

        return results, warning
