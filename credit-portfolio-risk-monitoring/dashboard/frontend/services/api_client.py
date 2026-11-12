# frontend/services/api_client.py

import os
from typing import Iterable, Optional

import httpx

_DEFAULT_API_BASES = (
    os.getenv("API_BASE") or os.getenv("API_BASE_URL") or "http://127.0.0.1:8000",
    "http://127.0.0.1:8050",
    "http://127.0.0.1:8000",
)


def _resolve_api_base():
    for base in _DEFAULT_API_BASES:
        base = base.rstrip("/")
        if not base:
            continue
        try:
            resp = httpx.get(f"{base}/overview/periods", timeout=2)
            if resp.status_code < 500:
                return base
        except Exception:
            continue
    return _DEFAULT_API_BASES[0].rstrip("/")


API_BASE = _resolve_api_base()


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
    resp = httpx.get(f"{API_BASE}/overview/summary", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_aggregates(period=None, loan_purpose=None):
    params = {}
    if period:
        params["period"] = period
    if loan_purpose:
        params["loan_purpose"] = _to_csv(loan_purpose)
    resp = httpx.get(f"{API_BASE}/overview/aggregates", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_options(period=None):
    params = {"period": period} if period else {}
    resp = httpx.get(f"{API_BASE}/overview/options", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_periods():
    resp = httpx.get(f"{API_BASE}/overview/periods", timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_expected_loss(period=None):
    params = {"period": period} if period else {}
    resp = httpx.get(f"{API_BASE}/risk/expected_loss", params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def get_var(period=None, confidence=0.99, nsim=5000, rho=0.05):
    params = {"confidence": confidence, "nsim": nsim, "rho": rho}
    if period:
        params["period"] = period
    resp = httpx.get(f"{API_BASE}/risk/var", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def get_survival(period=None, loan_purpose=None):
    params = {}
    if period:
        params["period"] = period
    if loan_purpose:
        params["loan_purpose"] = _to_csv(loan_purpose)
    resp = httpx.get(f"{API_BASE}/survival", params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()

def get_backtest():
    resp = httpx.get(f"{API_BASE}/risk/backtest", timeout=20)
    resp.raise_for_status()
    return resp.json()
