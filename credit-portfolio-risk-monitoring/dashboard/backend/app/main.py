"""Portfolio risk API: overview metrics, expected loss, credit VaR, survival and backtesting."""
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .services import insights, risk_metrics

logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Portfolio Risk API")

# Comma-separated list, e.g. "https://dashboard-credit-risk.onrender.com,http://localhost:8050"
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8050").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_methods=["GET"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/overview/summary")
def overview_summary(period: Optional[str] = None, loan_purpose: Optional[str] = None):
    """Return summary metrics for the requested period and optional loan_purpose filter.
    Query params:
    - period: e.g. '2018Q4'
    - loan_purpose: comma-separated list of loan purpose values
    """
    df = insights.load_data(period).copy()
    # df = insights.load_data(period)
    if df is None or df.empty:
        return {'total_lent': 0.0, 'repaid': 0.0, 'default_rate': 0.0, 'interest_earned': 0.0}
    if loan_purpose:
        vals = [v.strip() for v in loan_purpose.split(',') if v.strip()]
        # ensure loan_purpose column exists
        if 'loan_purpose' not in df.columns and 'purpose' in df.columns:
            df['loan_purpose'] = df['purpose']
        if 'loan_purpose' in df.columns:
            df = df[df['loan_purpose'].isin(vals)]
    return insights.summary(df)


@app.get("/overview/aggregates")
def overview_aggregates(period: Optional[str] = None, loan_purpose: Optional[str] = None):
    """Return grouped aggregates by loan purpose.
    Returns JSON: { data: [ {loan_purpose, lent, repaid, interest}, ... ] }
    """
    df = insights.load_data(period).copy()
    if df is None or df.empty:
        return {'data': []}
    if loan_purpose:
        vals = [v.strip() for v in loan_purpose.split(',') if v.strip()]
        if 'loan_purpose' not in df.columns and 'purpose' in df.columns:
            df['loan_purpose'] = df['purpose']
        if 'loan_purpose' in df.columns:
            df = df[df['loan_purpose'].isin(vals)]
    rows = insights.aggregates(df)
    return {'data': rows}


@app.get("/overview/options")
def overview_options(period: Optional[str] = None):
    """Return available loan purpose options for the period (or latest when omitted)."""
    df = insights.load_data(period).copy()
    if df is None or df.empty:
        return {'loan_purposes': []}
    opts = insights.loan_purposes(df)
    return {'loan_purposes': opts}


@app.get("/overview/periods")
def overview_periods():
    """Return available period identifiers (e.g. '2018Q4')."""
    try:
        periods = insights.list_periods().copy()
        return {'periods': periods}
    except Exception:
        return {'periods': []}



@app.get("/risk/expected_loss")
def risk_expected_loss(period: Optional[str] = None):
    """Return total expected loss and simple breakdown for the requested period."""
    try:
        df = insights.load_data(period).copy()
        if df is None or df.empty:
            return {'total_expected_loss': 0.0, 'n_loans': 0}
        total_el, _, by_grade = risk_metrics.expected_loss(df)
        return {
            'total_expected_loss': total_el,
            'n_loans': int(len(df)),
            'by_grade': by_grade.to_dict(orient='records'),
        }
    except Exception as e:
        logger.exception("expected_loss failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk/var")
def risk_var(period: Optional[str] = None, confidence: float = 0.99, nsim: int = 5000,
             rho: float = risk_metrics.DEFAULT_ASSET_CORRELATION):
    """Credit VaR from a one-factor Vasicek model.

    Query params:
    - period: quarter identifier
    - confidence: VaR confidence (0-1)
    - nsim: number of simulated economic scenarios
    - rho: asset correlation (0.03-0.16 is typical for retail unsecured loans)
    """
    try:
        df = insights.load_data(period).copy()
        if df is None or df.empty:
            return {'var': 0.0, 'expected_loss': 0.0, 'hist': {'bins': [], 'counts': []}}
        losses = risk_metrics.simulate_portfolio_losses(df, nsim=int(nsim), rho=float(rho))
        counts, bin_edges = np.histogram(losses, bins=50)
        return {
            'var': risk_metrics.var_from_simulations(losses, confidence=float(confidence)),
            'var_closed_form': risk_metrics.vasicek_var(df, confidence=float(confidence), rho=float(rho)),
            'expected_loss': float(losses.mean()),
            'confidence': float(confidence),
            'rho': float(rho),
            'hist': {'bins': bin_edges.tolist(), 'counts': counts.tolist()},
        }
    except Exception as e:
        logger.exception("var failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk/backtest")
def risk_backtest():
    """Expected vs realised loss and default rate for every vintage.

    The table is precomputed by scripts/prepare_data.py (loading every quarter
    at request time would exceed a free-tier instance's memory).
    """
    try:
        path = insights.DEFAULT_DATA_DIR / "backtest_by_vintage.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Run scripts/prepare_data.py to build the backtest table")
        table = pd.read_csv(path)
        return {'data': table.replace({np.nan: None}).to_dict(orient='records')}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("backtest failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/survival")
def survival(period: Optional[str] = None, loan_purpose: Optional[str] = None):
    """Return a simple Kaplan-Meier style survival curve for the cohort."""
    try:
        df = insights.load_data(period).copy()
        if df is None or df.empty:
            return {'months': [], 'survival': [], 'events': [], 'at_risk': [], 'median_months': None, 'n_loans': 0}
        if loan_purpose:
            vals = [v.strip() for v in loan_purpose.split(',') if v.strip()]
            if 'loan_purpose' not in df.columns and 'purpose' in df.columns:
                df['loan_purpose'] = df['purpose']
            if 'loan_purpose' in df.columns:
                df = df[df['loan_purpose'].isin(vals)]
        result = risk_metrics.survival(df)
        return result
    except Exception as e:
        return {'months': [], 'survival': [], 'events': [], 'at_risk': [], 'median_months': None, 'n_loans': 0, 'error': str(e)}
