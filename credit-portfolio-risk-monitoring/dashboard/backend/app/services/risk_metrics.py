"""Portfolio risk metrics: expected loss, survival analysis, and loss simulation.

Survival analysis section
--------------------------
Given a loan-level DataFrame with `issue_d`, `loan_status`, and (for defaulted
loans) `last_pymnt_d`, this module builds a duration/event-flag representation
suitable for time-to-default modelling, then exposes:

- `kaplan_meier_survival` : non-parametric baseline survival curve S(t)
- `cox_hazard`            : covariate-adjusted proportional hazards model
- `survival`              : convenience wrapper combining both

Censoring convention
---------------------
A loan is treated as an "event" (default) if its status matches a default-like
keyword (charged off, default, late, grace period, etc). For an event, the
observation ends at `last_pymnt_d` (the last point we know the loan was still
alive). For a non-event (current/paid-off) loan, the observation is
right-censored at "now" -- we only know it survived at least that long.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAYS_PER_MONTH = 30.4375  # average calendar month length, accounts for leap years

# Default definition: charged off, in default, or 31-120 days late.
# "In Grace Period" and "Late (16-30 days)" are early arrears, not default.
# Note: "Does not meet the credit policy. Status:Fully Paid" must NOT match,
# so we match on the status text after "Status:" when present.
DEFAULT_STATUS_KEYWORDS = (
    "charged off",
    "default",
    "late (31-120",
)

DEFAULT_COX_COVARIATES = (
    "fico_range_low",
    "dti",
    "funded_amnt",
    "annual_inc",
    "int_rate",
)

# Basel-style asset correlation for retail unsecured exposures sits roughly
# between 0.03 and 0.16; 0.05 is a reasonable default for consumer loans.
DEFAULT_ASSET_CORRELATION = 0.05


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def _finite_or_none(value: Any) -> Optional[float]:
    """Cast to float, returning None for NaN/inf/unparseable values."""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return fv if np.isfinite(fv) else None


def _finite_list(values: Sequence[Any]) -> List[Optional[float]]:
    """Element-wise `_finite_or_none` over an iterable."""
    return [_finite_or_none(v) for v in values]


def _empty_survival_result() -> Dict[str, Any]:
    """Shared shape for callers when there isn't enough data to fit a curve."""
    return {
        "months": [],
        "survival": [],
        "events": [],
        "at_risk": [],
        "median_months": None,
        "n_loans": 0,
    }


# ---------------------------------------------------------------------------
# duration / event construction (shared by KM, Cox, and the legacy fallback)
# ---------------------------------------------------------------------------

def is_default_status(status: Any) -> bool:
    """True if a loan_status value indicates a default-like event."""
    if pd.isna(status):
        return False
    value = str(status).strip().lower()
    if not value:
        return False
    # "Does not meet the credit policy. Status:Charged Off" -> "charged off"
    if "status:" in value:
        value = value.split("status:", 1)[1].strip()
    return any(keyword in value for keyword in DEFAULT_STATUS_KEYWORDS)


# Backwards-compatible alias: keep the original private name working in case
# other modules import it directly.
_default_event_flag = lambda status: int(is_default_status(status))


def _event_flags(df: pd.DataFrame) -> pd.Series:
    """1/0 event indicator per row, defaulting to "no event" if status is absent."""
    if "loan_status" not in df.columns:
        return pd.Series(0, index=df.index)
    return df["loan_status"].apply(is_default_status).astype(int)


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    """Parse to UTC then drop tz info, so mixed tz-aware/naive inputs compare cleanly."""
    # Lending Club stores dates as "Mar-2019"; parse that format first, then fall back.
    parsed = pd.to_datetime(series, format="%b-%Y", errors="coerce", utc=True)
    if parsed.isna().all() and series.notna().any():
        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_localize(None)


def snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """The data's as-of date: the latest payment date observed in the extract.

    Censoring at today's date would give a 2013 loan 13 years of exposure even
    though the data stops in 2019, so we censor at the extract date instead.
    """
    if "last_pymnt_d" in df.columns:
        latest = _to_naive_datetime(df["last_pymnt_d"]).max()
        if pd.notna(latest):
            return latest.normalize()
    return pd.Timestamp.utcnow().tz_localize(None).normalize()


def months_since_issue(X: pd.DataFrame) -> pd.Series:
    """Observation window, in months, for each loan.

    - Defaulted loans: issue_d -> last_pymnt_d (last date known alive).
    - Paid-off loans: issue_d -> last_pymnt_d (they leave the risk set at payoff).
    - Still-active loans: issue_d -> snapshot date of the extract.
    Returns 0 for rows that can't be parsed so the output aligns 1:1 with input.
    """
    df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if df.empty or "issue_d" not in df.columns:
        return pd.Series(dtype=float, index=df.index)

    issue_d = _to_naive_datetime(df["issue_d"])
    reference_dates = pd.Series(snapshot_date(df), index=df.index)

    if "last_pymnt_d" in df.columns:
        last_pymnt_d = _to_naive_datetime(df["last_pymnt_d"])
        status = df["loan_status"].astype(str).str.lower() if "loan_status" in df.columns else pd.Series("", index=df.index)
        closed = _event_flags(df).astype(bool) | status.str.contains("fully paid")
        use_last_payment = closed & last_pymnt_d.notna()
        reference_dates = reference_dates.mask(use_last_payment, last_pymnt_d)

    months = ((reference_dates - issue_d).dt.days / DAYS_PER_MONTH).clip(lower=0)
    return months.fillna(0.0)


@dataclass
class _DurationFrame:
    """Loan-level duration + event data, ready for KM/Cox fitting."""

    duration: pd.Series
    event: pd.Series
    n_loans: int
    frame: pd.DataFrame = field(repr=False)


def _build_duration_frame(X: pd.DataFrame) -> Optional[_DurationFrame]:
    """Parse issue_d, drop unparseable rows, and compute duration/event columns.

    Returns None if there isn't enough information to build a curve at all.
    """
    if X is None or X.empty or "issue_d" not in X.columns:
        return None

    df = X.copy()
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df = df.dropna(subset=["issue_d"])
    if df.empty:
        return None

    duration = months_since_issue(df)
    event = _event_flags(df)
    return _DurationFrame(duration=duration, event=event, n_loans=len(df), frame=df)


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------

def _legacy_survival_curve(dframe: _DurationFrame) -> Dict[str, Any]:
    """Manual Kaplan-Meier calculation, used only if `lifelines` isn't installed."""
    month_index = np.ceil(dframe.duration).astype(int)
    months = sorted(month_index.unique().tolist())

    survival_curve, events_list, at_risk_list = [], [], []
    running_survival = 1.0

    for month in months:
        at_risk = int((month_index >= month).sum())
        if at_risk == 0:
            continue
        events = int(((month_index == month) & (dframe.event == 1)).sum())
        if events > 0:
            running_survival *= 1 - (events / at_risk)
        survival_curve.append(float(running_survival))
        events_list.append(events)
        at_risk_list.append(at_risk)

    median_months = next(
        (float(m) for m, s in zip(months, survival_curve) if s <= 0.5), None
    )

    return {
        "months": [int(m) for m in months],
        "survival": survival_curve,
        "events": events_list,
        "at_risk": at_risk_list,
        "median_months": median_months,
        "n_loans": dframe.n_loans,
    }


def _km_confidence_interval(kmf) -> tuple:
    """Extract (lower, upper) CI bounds from a fitted KaplanMeierFitter, if available."""
    try:
        ci = kmf.confidence_interval_
    except Exception:
        return None, None

    if ci.empty:
        return None, None

    cols = ci.columns
    lower = _finite_list(ci.iloc[:, 0].tolist())
    upper = _finite_list(ci.iloc[:, 1].tolist()) if len(cols) >= 2 else list(lower)
    return lower, upper


def kaplan_meier_survival(X: pd.DataFrame) -> Dict[str, Any]:
    """Non-parametric baseline survival curve S(t) for the given cohort.

    Falls back to a manual KM calculation if `lifelines` is not installed.
    """
    dframe = _build_duration_frame(X)
    if dframe is None:
        return _empty_survival_result()

    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return _legacy_survival_curve(dframe)

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=dframe.duration.to_numpy(),
        event_observed=dframe.event.to_numpy(),
        label="Portfolio",
    )

    survival_table = kmf.survival_function_
    event_table = kmf.event_table
    lower_ci, upper_ci = _km_confidence_interval(kmf)

    median = kmf.median_survival_time_
    return {
        "months": _finite_list(survival_table.index.tolist()),
        "survival": _finite_list(survival_table.iloc[:, 0].tolist()),
        "events": event_table["observed"].astype(int).tolist(),
        "at_risk": event_table["at_risk"].astype(int).tolist(),
        "median_months": _finite_or_none(median) if pd.notna(median) else None,
        "n_loans": dframe.n_loans,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
    }


# ---------------------------------------------------------------------------
# Cox proportional hazards
# ---------------------------------------------------------------------------

def _cox_average_survival_curve(cph, cov_df: pd.DataFrame) -> Optional[Dict[str, List[Optional[float]]]]:
    """Cohort-average predicted survival curve, for plotting alongside KM."""
    if cov_df.empty:
        return None
    try:
        predicted = cph.predict_survival_function(cov_df)
        mean_survival = predicted.mean(axis=1)
    except Exception:
        logger.warning("Cox survival-curve prediction failed", exc_info=True)
        return None

    return {
        "months": _finite_list(mean_survival.index.tolist()),
        "survival": _finite_list(mean_survival.values.tolist()),
    }


def cox_hazard(X: pd.DataFrame, covariates: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Fit a Cox proportional hazards model and return hazard ratios + summary.

    `covariates` defaults to whichever of DEFAULT_COX_COVARIATES are present
    in `X`. Rows with missing values in duration/event/covariates are dropped
    (Cox fitting requires complete cases).
    """
    empty_result = {"hazard_ratios": {}, "summary": None, "n_loans": 0}

    dframe = _build_duration_frame(X)
    if dframe is None:
        return empty_result

    covariates = list(covariates) if covariates is not None else [
        col for col in DEFAULT_COX_COVARIATES if col in dframe.frame.columns
    ]

    cph_df = dframe.frame.assign(duration=dframe.duration, event=dframe.event)
    cph_df = cph_df[["duration", "event", *covariates]].dropna()

    if len(cph_df) < 2:
        return {**empty_result, "n_loans": dframe.n_loans}

    try:
        from lifelines import CoxPHFitter
    except ImportError:
        return {
            **empty_result,
            "n_loans": dframe.n_loans,
            "fallback": "lifelines not installed; Cox model unavailable",
        }

    cph = CoxPHFitter()
    cph.fit(cph_df, duration_col="duration", event_col="event")

    return {
        "hazard_ratios": cph.hazard_ratios_.to_dict(),
        "summary": cph.summary.to_dict(),
        "n_loans": len(cph_df),
        "survival_curve": _cox_average_survival_curve(cph, cph_df[covariates]),
    }


def survival(X: pd.DataFrame) -> Dict[str, Any]:
    """Kaplan-Meier curve for the cohort, enriched with Cox output when available.

    The Cox fit is best-effort: if it fails (e.g. too few events, collinear
    covariates), the KM curve is still returned on its own.
    """
    result = kaplan_meier_survival(X)

    try:
        cox_out = cox_hazard(X)
    except Exception:
        logger.warning("Cox hazard model failed; returning KM curve only", exc_info=True)
        return result

    if cox_out.get("survival_curve"):
        result["cox_survival"] = cox_out["survival_curve"]
        result["cox_hazard_ratios"] = cox_out.get("hazard_ratios", {})

    return result


# ---------------------------------------------------------------------------
# Expected loss: EL = PD x LGD x EAD
# ---------------------------------------------------------------------------
#
# PD  : calibrated lifetime probability of default (`pd_calibrated`, written by
#       scripts/prepare_data.py). Raw model scores are NOT probabilities.
# LGD : share of the exposure lost after recoveries, estimated per grade from
#       loans that actually defaulted.
# EAD : share of the funded amount still outstanding at default, estimated per
#       grade from defaulted loans (a simple credit-conversion factor).

def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def pd_column(df: pd.DataFrame) -> pd.Series:
    """Calibrated PD if available; otherwise the raw score, with a warning."""
    if "pd_calibrated" in df.columns:
        return _num(df, "pd_calibrated").clip(0.0, 1.0)
    logger.warning("pd_calibrated missing: using raw prob_default, which is not a calibrated probability")
    return _num(df, "prob_default").clip(0.0, 1.0)


def realised_loss(df: pd.DataFrame) -> pd.Series:
    """Principal lost on defaulted loans, net of recoveries (0 for non-defaults)."""
    lost = (_num(df, "funded_amnt") - _num(df, "total_rec_prncp")
            - _num(df, "recoveries") + _num(df, "collection_recovery_fee")).clip(lower=0.0)
    return lost.where(_event_flags(df).astype(bool), 0.0)


def lgd_ead_by_grade(df: pd.DataFrame, group_col: str = "grade") -> pd.DataFrame:
    """Empirical LGD and EAD factor per grade, from loans that defaulted."""
    defaulted = df[_event_flags(df).astype(bool)].copy()
    if defaulted.empty:
        return pd.DataFrame(columns=[group_col, "lgd", "ead_factor", "n_defaults"])
    funded = _num(defaulted, "funded_amnt").replace(0, np.nan)
    exposure = (funded - _num(defaulted, "total_rec_prncp")).clip(lower=0.0)
    net_recovery = _num(defaulted, "recoveries") - _num(defaulted, "collection_recovery_fee")
    defaulted["ead_factor"] = (exposure / funded).clip(0.0, 1.0)
    defaulted["lgd"] = (1.0 - net_recovery / exposure.replace(0, np.nan)).clip(0.0, 1.0)
    key = group_col if group_col in defaulted.columns else None
    grouped = defaulted.groupby(key, observed=True) if key else defaulted.assign(_all="all").groupby("_all")
    out = grouped.agg(lgd=("lgd", "mean"), ead_factor=("ead_factor", "mean"),
                      n_defaults=("lgd", "size")).reset_index()
    return out.rename(columns={"_all": group_col})


def _attach_lgd_ead(df: pd.DataFrame, group_col: str = "grade") -> pd.DataFrame:
    params = lgd_ead_by_grade(df, group_col)
    out = df.copy()
    if group_col in out.columns and not params.empty:
        out = out.merge(params[[group_col, "lgd", "ead_factor"]], on=group_col, how="left")
    overall = lgd_ead_by_grade(df.drop(columns=[group_col], errors="ignore"), group_col)
    fallback_lgd = float(overall["lgd"].iloc[0]) if not overall.empty else 0.6
    fallback_ead = float(overall["ead_factor"].iloc[0]) if not overall.empty else 0.6
    out["lgd"] = out.get("lgd", pd.Series(np.nan, index=out.index)).fillna(fallback_lgd)
    out["ead_factor"] = out.get("ead_factor", pd.Series(np.nan, index=out.index)).fillna(fallback_ead)
    out.index = df.index
    return out


def expected_loss(X: pd.DataFrame):
    """Lifetime expected loss at origination: PD x LGD x (EAD factor x funded amount).

    Returns (total EL, per-loan EL series, per-grade table).
    """
    df = _attach_lgd_ead(X)
    ead = df["ead_factor"] * _num(df, "funded_amnt")
    el = pd_column(df) * df["lgd"] * ead
    by_grade = pd.DataFrame()
    if "grade" in df.columns:
        by_grade = (df.assign(el=el, ead=ead, pd=pd_column(df))
                      .groupby("grade", observed=True)
                      .agg(n_loans=("el", "size"), expected_loss=("el", "sum"),
                           avg_pd=("pd", "mean"), lgd=("lgd", "first"))
                      .reset_index())
    return float(el.sum()), el, by_grade


def backtest_by_vintage(X: pd.DataFrame) -> pd.DataFrame:
    """Expected vs realised loss and default rate per issue quarter.

    Only fully matured vintages give a fair comparison, so the share of loans
    still active is returned alongside each row.
    """
    df = X.copy()
    if df.empty or "issue_d" not in df.columns:
        return pd.DataFrame()
    _, el, _ = expected_loss(df)
    status = df["loan_status"].astype(str).str.lower() if "loan_status" in df.columns else pd.Series("", index=df.index)
    df = df.assign(
        vintage=pd.to_datetime(df["issue_d"], errors="coerce").dt.to_period("Q").astype(str),
        el=el, realised=realised_loss(df), pd=pd_column(df),
        defaulted=_event_flags(df), active=status.str.contains("current|grace|late \\(16"),
    )
    out = (df.groupby("vintage", observed=True)
             .agg(n_loans=("el", "size"), expected_loss=("el", "sum"), realised_loss=("realised", "sum"),
                  predicted_default_rate=("pd", "mean"), observed_default_rate=("defaulted", "mean"),
                  share_active=("active", "mean"))
             .reset_index())
    out["loss_ratio"] = out["realised_loss"] / out["expected_loss"].replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------
# Credit VaR: one-factor Vasicek / Gaussian copula
# ---------------------------------------------------------------------------
#
# Defaults are NOT independent: a weak economy pushes many borrowers into
# default at once. In the one-factor model each borrower's credit quality is
#   A_i = sqrt(rho) * Z + sqrt(1 - rho) * e_i
# where Z is the shared economic factor. Conditional on Z, the default
# probability is
#   p_i(Z) = Phi( (Phi^-1(PD_i) - sqrt(rho) * Z) / sqrt(1 - rho) )
# For a large, granular loan book, the loss given Z is close to its
# conditional expectation sum_i p_i(Z) * LGD_i * EAD_i (the ASRF assumption
# behind Basel IRB), so we simulate Z only.

def _norm():
    from scipy.stats import norm  # scipy ships with scikit-learn
    return norm


def _pd_exposure_buckets(df: pd.DataFrame, n_buckets: int = 200):
    """Group loans into PD buckets so simulation cost doesn't grow with book size.

    Loss is linear in exposure, so summing LGD x EAD within narrow PD buckets
    gives practically the same distribution as loan-by-loan simulation.
    """
    frame = _attach_lgd_ead(df)
    pd_i = pd_column(frame).clip(1e-6, 1 - 1e-6)
    exposure = frame["lgd"] * frame["ead_factor"] * _num(frame, "funded_amnt")
    buckets = pd.qcut(pd_i.rank(method="first"), q=min(n_buckets, len(pd_i)), labels=False)
    grouped = pd.DataFrame({"pd": pd_i, "w": exposure, "b": buckets}).groupby("b")
    weight = grouped["w"].sum()
    weighted_pd = (pd.DataFrame({"pd": pd_i * exposure, "b": buckets}).groupby("b")["pd"].sum())
    mean_pd = (weighted_pd / weight.replace(0, np.nan)).fillna(grouped["pd"].mean())
    return mean_pd.values, weight.values


def simulate_portfolio_losses(df: pd.DataFrame, nsim: int = 5000, rho: float = DEFAULT_ASSET_CORRELATION,
                              seed: Optional[int] = 42, batch_size: int = 250) -> np.ndarray:
    """Simulated portfolio loss distribution under the one-factor model."""
    if df is None or df.empty:
        return np.array([])
    norm = _norm()
    pd_i, exposure = _pd_exposure_buckets(df)
    threshold = norm.ppf(pd_i)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(nsim)
    losses = np.empty(nsim)
    for start in range(0, nsim, batch_size):
        zb = z[start:start + batch_size, None]
        cond_pd = norm.cdf((threshold[None, :] - np.sqrt(rho) * zb) / np.sqrt(1 - rho))
        losses[start:start + batch_size] = cond_pd @ exposure
    return losses


def vasicek_var(df: pd.DataFrame, confidence: float = 0.99, rho: float = DEFAULT_ASSET_CORRELATION) -> float:
    """Closed-form loss quantile of the one-factor model (no simulation needed)."""
    if df is None or df.empty:
        return 0.0
    norm = _norm()
    pd_i, exposure = _pd_exposure_buckets(df)
    stressed = norm.cdf((norm.ppf(pd_i) + np.sqrt(rho) * norm.ppf(confidence)) / np.sqrt(1 - rho))
    return float(stressed @ exposure)


def var_from_simulations(losses: np.ndarray, confidence: float = 0.95) -> float:
    """Portfolio VaR at the given confidence level from simulated losses."""
    if losses is None or len(losses) == 0:
        return 0.0
    return float(np.quantile(losses, confidence))