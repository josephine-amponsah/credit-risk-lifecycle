"""Train an origination-time probability-of-default (PD) model on Lending Club loans.

Design choices (each one fixes a problem in the original notebook):
- Origination features only. Anything recorded after the loan was issued
  (outstanding principal, payment dates, recoveries, hardship or settlement
  fields, latest FICO) leaks the outcome and is excluded.
- Binary target on matured loans only: 1 = Charged Off / Default /
  Late (31-120 days), 0 = Fully Paid. Loans still running have no outcome yet.
- Out-of-time validation: train on earlier vintages, test on later ones,
  the way the model would actually be used.
- No class rebalancing, so predicted probabilities stay close to real default
  rates (calibration is checked and refined in prepare_data.py).

Usage
    python scripts/train_pd_model.py --src notebooks/data_splits \
        --scored-dst build/scored --train-end 2014Q4 --test-end 2015Q4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

NUMERIC = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs", "fico_range_low",
    "inq_last_6mths", "mths_since_last_delinq", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies", "acc_open_past_24mths", "bc_util",
    "num_actv_rev_tl", "tot_cur_bal", "total_rev_hi_lim", "avg_cur_bal", "percent_bc_gt_75",
    "term_months", "emp_years", "credit_history_months",
]
CATEGORICAL = ["sub_grade", "home_ownership", "verification_status", "purpose", "application_type"]
DEFAULT_STATUSES = {"Charged Off", "Default", "Late (31-120 days)"}


# Raw columns needed to build features, plus the ones the dashboard keeps.
RAW_COLUMNS = sorted(set(
    [c for c in NUMERIC if c not in ("term_months", "emp_years", "credit_history_months")]
    + CATEGORICAL
    + ["term", "emp_length", "earliest_cr_line", "issue_d", "loan_status"]
    + ["id", "last_pymnt_d", "grade", "funded_amnt", "out_prncp", "total_rec_prncp", "total_rec_int",
       "recoveries", "collection_recovery_fee", "addr_state"]
))


def load(src: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    frames = []
    for f in sorted(src.glob("data_*.parquet")):
        available = set(pq.read_schema(f).names)
        part = pd.read_parquet(f, columns=[c for c in RAW_COLUMNS if c in available])
        for col in part.select_dtypes(include="object").columns:
            if col != "id":
                part[col] = part[col].astype("category")  # keeps 2M+ rows in memory
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df  # modified in place to save memory
    out["issue_d"] = pd.to_datetime(out["issue_d"], errors="coerce")
    out["vintage"] = out["issue_d"].dt.to_period("Q")
    out["term_months"] = pd.to_numeric(out["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    emp = out["emp_length"].astype(str).str.replace("< 1", "0", regex=False)
    out["emp_years"] = pd.to_numeric(emp.str.extract(r"(\d+)")[0], errors="coerce")
    earliest = pd.to_datetime(out["earliest_cr_line"], errors="coerce", format="mixed")
    out["credit_history_months"] = (out["issue_d"] - earliest).dt.days / 30.4375
    for col in NUMERIC:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].astype(str).str.rstrip("%"), errors="coerce")
    for col in CATEGORICAL:
        out[col] = out[col].astype("category")
    status = out["loan_status"].astype(str).str.split("Status:").str[-1].str.strip()
    out["target"] = np.where(status.isin(DEFAULT_STATUSES), 1, np.where(status == "Fully Paid", 0, -1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=ROOT / "notebooks" / "data_splits")
    parser.add_argument("--scored-dst", type=Path, default=ROOT / "build" / "scored")
    parser.add_argument("--model-dst", type=Path, default=ROOT / "models")
    parser.add_argument("--train-end", default="2014Q4")
    parser.add_argument("--test-end", default="2015Q4")
    args = parser.parse_args()

    df = engineer(load(args.src))
    features = NUMERIC + CATEGORICAL
    matured = df["target"] >= 0
    train = df[matured & (df["vintage"] <= pd.Period(args.train_end, "Q"))]
    test = df[matured & (df["vintage"] > pd.Period(args.train_end, "Q"))
              & (df["vintage"] <= pd.Period(args.test_end, "Q"))]

    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, categorical_features="from_dtype",
        early_stopping=True, validation_fraction=0.1, random_state=42,
    )
    model.fit(train[features], train["target"])

    p_test = model.predict_proba(test[features])[:, 1]
    auc = roc_auc_score(test["target"], p_test)
    metrics = {
        "train_vintages": f"<= {args.train_end}", "test_vintages": f"> {args.train_end} and <= {args.test_end}",
        "n_train": int(len(train)), "n_test": int(len(test)),
        "test_auc": round(float(auc), 4), "test_gini": round(float(2 * auc - 1), 4),
        "test_brier": round(float(brier_score_loss(test["target"], p_test)), 4),
        "test_default_rate": round(float(test["target"].mean()), 4),
        "test_mean_pd": round(float(p_test.mean()), 4),
    }
    print(json.dumps(metrics, indent=2))

    args.model_dst.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features}, args.model_dst / "pd_model.joblib", compress=3)
    (args.model_dst / "pd_model_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Score every loan with the new model; prepare_data.py then calibrates it.
    args.scored_dst.mkdir(parents=True, exist_ok=True)
    df["prob_default"] = model.predict_proba(df[features])[:, 1]
    df["quarter"] = df["vintage"].astype(str)
    keep = [c for c in RAW_COLUMNS if c in df.columns] + ["prob_default", "quarter"]
    for quarter, part in df[keep].groupby("quarter"):
        part.to_parquet(args.scored_dst / f"data_{quarter}.parquet", index=False)


if __name__ == "__main__":
    main()
