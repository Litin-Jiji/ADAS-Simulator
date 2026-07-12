"""
tests/test_collision_risk.py
Unit tests for collision risk assessor.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from services.collision_risk import CollisionRiskAssessor, RISK_CRITICAL, RISK_LOW


def _make_obj(tid, dist):
    return {
        "track_id":   tid,
        "class_name": "car",
        "bbox":       [100, 100, 200, 200],
        "centroid":   (150, 150),
        "distance_m": dist,
    }


def test_low_risk_far_object():
    assessor = CollisionRiskAssessor()
    tracked  = [_make_obj(1, 25.0)]
    results, warning = assessor.assess(tracked)
    assert warning.risk == RISK_LOW


def test_critical_risk_very_close():
    assessor = CollisionRiskAssessor()
    # Feed multiple frames so closing speed builds up
    for dist in [8.0, 6.0, 4.0, 2.5, 1.5]:
        tracked  = [_make_obj(1, dist)]
        results, warning = assessor.assess(tracked)
    assert warning.risk == RISK_CRITICAL


def test_no_objects():
    assessor = CollisionRiskAssessor()
    results, warning = assessor.assess([])
    assert len(results) == 0
    assert warning.active is False


def test_multiple_objects_worst_wins():
    assessor = CollisionRiskAssessor()
    tracked = [_make_obj(1, 20.0), _make_obj(2, 2.0)]
    results, warning = assessor.assess(tracked)
    assert warning.risk == RISK_CRITICAL
