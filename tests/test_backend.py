"""
tests/test_backend.py
Basic unit tests for the FastAPI backend.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_status_not_running():
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert "running" in data
    assert data["running"] is False


def test_analytics_empty():
    r = client.get("/api/analytics")
    assert r.status_code == 200
    data = r.json()
    assert "total_vehicles" in data
    assert "near_misses" in data


def test_stop_when_not_running():
    r = client.post("/api/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "stopped"
# erqafc