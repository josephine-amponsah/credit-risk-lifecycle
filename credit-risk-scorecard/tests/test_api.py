"""Smoke tests for the scorecard API (needs torch + shap installed)."""
import json
import runpy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app, compute_risk_score

client = TestClient(app)
USERS = json.loads((Path(__file__).resolve().parents[1] / "api" / "sample_users.json").read_text())


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_users_endpoint_lists_sample_users():
    assert len(client.get("/users").json()["users"]) == len(USERS)


@pytest.mark.parametrize("user", USERS[:5], ids=lambda u: u["user_id"])
def test_score_is_valid(user):
    body = client.post("/score", json=user).json()
    assert 300 <= body["risk_score"] <= 850
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["risk_band"] in {"Poor", "Fair", "Good", "Excellent"}


def test_even_odds_scores_600():
    assert compute_risk_score([0.3, 0.4, 0.3]) == 600


def test_main_module_starts_uvicorn_when_executed(monkeypatch):
    called = {}

    def fake_run(app, host, port):
        called["app"] = app
        called["host"] = host
        called["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "api" / "main.py"), run_name="__main__")

    assert called["app"].title == app.title
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 8001
