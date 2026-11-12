"""Build the dashboard's data from the scored Lending Club quarters.

Steps
1. Keep only the columns the dashboard uses (153 -> ~20 columns, ~90% smaller).
2. Calibrate the model score into a probability of default (PD) with isotonic
   regression, fitted on matured early vintages only, then applied to every
   vintage. Later vintages are therefore an out-of-time test.
3. Precompute the expected-vs-realised backtest table per vintage.

Usage
    python scripts/prepare_data.py --src notebooks/data_splits \
        --dst dashboard/backend/app/data --calibration-end 2014Q4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard" / "backend"))
from app.services import risk_metrics  # noqa: E402

KEEP_COLS = [
    "id", "issue_d", "quarter", "loan_status", "last_pymnt_d", "term", "grade", "sub_grade",
    "purpose", "int_rate", "funded_amnt", "out_prncp", "total_rec_prncp", "total_rec_int",
    "recoveries", "collection_recovery_fee", "annual_inc", "dti", "fico_range_low",
    "addr_state", "prob_default",
]


def load_quarters(src: Path) -> pd.DataFrame:
    files = sorted(src.glob("data_*.parquet"))
    if not files:
        raise SystemExit(f"No data_*.parquet files found in {src}")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        frames.append(df[[c for c in KEEP_COLS if c in df.columns]])
    return pd.concat(frames, ignore_index=True)


def fit_calibrator(df: pd.DataFrame, calibration_end: str) -> IsotonicRegression:
    vintage = pd.to_datetime(df["issue_d"]).dt.to_period("Q")
    status = df["loan_status"].astype(str).str.lower()
    matured = status.str.contains("fully paid") | risk_metrics._event_flags(df).astype(bool)
    train = df[(vintage <= pd.Period(calibration_end, "Q")) & matured]
    y = risk_metrics._event_flags(train)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(train["prob_default"].astype(float), y)
    print(f"Calibrator fitted on {len(train):,} matured loans up to {calibration_end}; "
          f"observed default rate {y.mean():.1%}, mean raw score {train['prob_default'].mean():.1%}")
    return iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=ROOT / "notebooks" / "data_splits")
    parser.add_argument("--dst", type=Path, default=ROOT / "dashboard" / "backend" / "app" / "data")
    parser.add_argument("--calibration-end", default="2014Q4")
    args = parser.parse_args()

    df = load_quarters(args.src)
    iso = fit_calibrator(df, args.calibration_end)
    df["pd_calibrated"] = iso.predict(df["prob_default"].astype(float))

    args.dst.mkdir(parents=True, exist_ok=True)
    df["quarter"] = pd.to_datetime(df["issue_d"]).dt.to_period("Q").astype(str)
    for quarter, part in df.groupby("quarter"):
        part.to_parquet(args.dst / f"data_{quarter}.parquet", index=False, compression="zstd")

    backtest = risk_metrics.backtest_by_vintage(df)
    backtest["in_calibration_sample"] = backtest["vintage"] <= args.calibration_end
    backtest.to_csv(args.dst / "backtest_by_vintage.csv", index=False)
    print(backtest.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
