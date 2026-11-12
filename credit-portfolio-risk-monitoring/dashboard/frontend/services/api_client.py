# frontend/services/api_client.py

import os
from typing import Iterable, Optional

import httpx

API_BASE = (os.getenv("API_BASE_URL") or os.getenv("API_BASE") or "http://127.0.0.1:8000").rstrip("/")

# One shared client: connection reuse, 3 retries on connection errors, and a timeout long enough
# for a Render free instance waking from sleep (about 50 seconds).
_client = httpx.Client(
    base_url=API_BASE,
    timeout=httpx.Timeout(90.0, connect=60.0),
    transport=httpx.HTTPTransport(retries=3),
)


def _get(path, params=None):
    resp = _client.get(path, params=params)
    resp.raise_for_status()
    return resp.json()


def warm_up():
    """Ping the backend so it starts waking while the page loads. Never raises."""
    try:
        _client.get("/health", timeout=5)
    except Exception:
        pass


def _to_csv(values):
    if not values:
        return None
    if isinstance(values, (list, tuple, set)):
        return ",".join(str(v).strip() for v in values if str(v).strip())
    return str(values)


def get_summary(period=None, loan_purpose=None):
    params = {}
    if period:
        params["period"] = period
    if loan_purpose:
        params["loan_purpose"] = _to_csv(loan_purpose)
    resp = _client.get("/overview/summary", params=params)
    resp.raise_for_status()
    return resp.json()


def get_aggregates(period=None, loan_purpose=None):
    params = {}
    if period:
        params["period"] = period
    if loan_purpose:
        params["loan_purpose"] = _to_csv(loan_purpose)
    resp = _client.get("/overview/aggregates", params=params)
    resp.raise_for_status()
    return resp.json()


def get_options(period=None):
    params = {"period": period} if period else {}
    resp = _client.get("/overview/options", params=params)
    resp.raise_for_status()
    return resp.json()


def get_periods():
    resp = _client.get("/overview/periods")
    resp.raise_for_status()
    return resp.json()


def get_expected_loss(period=None):
    params = {"period": period} if period else {}
    resp = _client.get("/risk/expected_loss", params=params)
    resp.raise_for_status()
    return resp.json()


def get_var(period=None, confidence=0.99, nsim=5000, rho=0.05):
    params = {"confidence": confidence, "nsim": nsim, "rho": rho}
    if period:
        params["period"] = period
    resp = _client.get("/risk/var", params=params)
    resp.raise_for_status()
    return resp.json()


def get_survival(period=None, loan_purpose=None):
    params = {}
    if period:
        params["period"] = period
    if loan_purpose:
        params["loan_purpose"] = _to_csv(loan_purpose)
    resp = _client.get("/survival", params=params)
    resp.raise_for_status()
    return resp.json()

def get_backtest():
    resp = _client.get("/risk/backtest")
    resp.raise_for_status()
    return resp.json()
