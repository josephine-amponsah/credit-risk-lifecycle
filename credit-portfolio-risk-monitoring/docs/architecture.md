# Architecture

## Data and model pipeline (offline)

```mermaid
flowchart TD
    raw[(Lending Club loans<br/>2013–2018, quarterly parquet)] --> train[train_pd_model.py<br/>origination features · out-of-time split]
    train --> model[[models/pd_model.joblib<br/>+ metrics json]]
    train --> scored[(build/scored)]
    scored --> prep[prepare_data.py<br/>isotonic calibration · column pruning]
    prep --> data[(backend/app/data<br/>24 slim parquet files)]
    prep --> bt[(backtest_by_vintage.csv)]
```

## Serving (online)

```mermaid
flowchart LR
    user((Risk analyst)) --> dash[Dash frontend<br/>:8050]
    dash -->|HTTP JSON| api[FastAPI backend<br/>:8000]
    api --> svc[services/<br/>insights · risk_metrics]
    svc --> data[(parquet data)]
```

| Component | Technology | Hosting |
|---|---|---|
| Frontend | Dash, Plotly, Dash Bootstrap | Render free web service |
| Backend | FastAPI, pandas, SciPy, lifelines | Render free web service |
| Data | Parquet, zstd-compressed (~75 MB) | committed with the backend |
| CI | GitHub Actions: tests, lint, Docker builds | GitHub |

## Design decisions

- **Heavy work happens offline.** Training, scoring and the vintage backtest run in scripts, so the API stays within a free instance's 512 MB of memory.
- **PD buckets for VaR.** Loans are grouped into 200 PD buckets before simulation, so VaR takes about a second for 100,000+ loans with no loss of accuracy (loss is linear in exposure).
- **A closed-form check.** Every simulated VaR is returned with its closed-form Vasicek value, so a broken simulation shows up immediately.
- **Fail loudly.** Endpoints return HTTP 500 with the error instead of silently returning zeros.

## Scaling path (not needed for the demo)

- Move data to object storage (S3 or GCS) and query it with DuckDB.
- Schedule `prepare_data.py` monthly (GitHub Actions cron or Airflow) and alert when the backtest loss ratio or PSI drifts.
- Track model versions and metrics with MLflow.
