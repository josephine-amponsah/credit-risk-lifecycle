# Credit Portfolio Risk Monitoring

**PD model · Expected loss · Credit VaR · Survival analysis · Backtesting · FastAPI + Dash**

A monitoring system for a consumer loan book: how much the portfolio is expected to lose, how bad losses could get in a downturn, when loans tend to default, and whether the PD model still matches reality. Built on 2.2 million Lending Club loans issued 2013–2018.

**[Live dashboard](https://dashboard-credit-risk.onrender.com)** (free hosting: the first load can take about 30 seconds)

<!-- TODO: add 3 screenshots to docs/img/ and uncomment
![Overview](docs/img/overview.png) ![Credit VaR](docs/img/var.png) ![Backtesting](docs/img/backtest.png)
-->

## Results at a glance

| | |
|---|---|
| **PD model, out-of-time** | AUC **0.741** · Gini **0.48** · Brier 0.143 (trained on 2013–2014 vintages, tested on 2015) |
| **Calibration** | Predicted default rates within about **1 pp** of observed rates on matured vintages, including out-of-time 2015 vintages |
| **Expected loss, 2015Q4 vintage** | **$244m** on $1.97bn lent (12.4%); realised losses to date $230m |
| **Credit VaR, 2015Q4, 99%** | **$411m** at asset correlation ρ = 0.05; **$557m** at ρ = 0.15 |

## Methodology

### 1. Probability of default (PD)
`scripts/train_pd_model.py`

- **Origination features only.** Anything recorded after the loan was issued is excluded: outstanding principal, payment dates, recoveries, hardship and settlement fields, latest FICO. Using them leaks the outcome.
- **Binary target on matured loans:** Charged Off, Default or Late (31–120 days) = 1; Fully Paid = 0. Loans still running have no outcome yet and are excluded from training.
- **Out-of-time validation:** train on vintages up to 2014Q4, test on 2015, as the model would be used in practice.
- Gradient-boosted trees (`HistGradientBoostingClassifier`) with native categorical support. No class rebalancing, so probabilities stay close to true default rates.

### 2. Calibration
`scripts/prepare_data.py`

Isotonic regression maps model scores to probabilities, fitted on matured 2013–2014 loans only. Later vintages are an out-of-time test of the calibration.

### 3. Expected loss: EL = PD × LGD × EAD

| Component | Estimate |
|---|---|
| **PD** | calibrated lifetime probability of default |
| **LGD** | 1 − net recoveries ÷ exposure at default, per grade, from defaulted loans (about 90% for this unsecured book) |
| **EAD** | share of the funded amount still outstanding at default, per grade |

### 4. Credit VaR: one-factor Vasicek model

Defaults are correlated: a weak economy pushes many borrowers into default together. Each borrower's credit quality is driven by a shared economic factor *Z* and an individual shock:

```
A_i = √ρ · Z + √(1 − ρ) · ε_i
PD_i(Z) = Φ( (Φ⁻¹(PD_i) − √ρ · Z) / √(1 − ρ) )
```

The dashboard simulates 5,000 economic scenarios and reports the 99% loss quantile. A closed-form Vasicek quantile, the asymptotic single risk factor (ASRF) result behind Basel IRB, is computed alongside as a check; the two agree within 1%. The ρ slider shows how sensitive tail risk is to correlation.

### 5. Survival analysis

Kaplan–Meier and Cox proportional hazards curves of time to default. Defaulted loans end at their last payment date, paid-off loans are censored at payoff, and active loans are censored at the data extract date.

### 6. Backtesting

Predicted vs observed default rates and expected vs realised loss per vintage. Recent vintages are still running, so the share of active loans is shown next to each row.

## Architecture

FastAPI serves the risk metrics; a multi-page Dash app calls it. Data preparation runs offline, so the API only reads small parquet files. See [docs/architecture.md](docs/architecture.md).

```
scripts/train_pd_model.py  →  scripts/prepare_data.py  →  dashboard/backend/app/data/*.parquet
                                                              ↓
                                            FastAPI (dashboard/backend)  ←  Dash (dashboard/frontend)
```

| Endpoint | Returns |
|---|---|
| `GET /overview/summary` | lent, repaid, default rate, interest for a vintage |
| `GET /risk/expected_loss` | total EL and EL by grade |
| `GET /risk/var?confidence=0.99&rho=0.05` | simulated and closed-form credit VaR, loss histogram |
| `GET /survival` | Kaplan–Meier curve, Cox hazard ratios |
| `GET /risk/backtest` | predicted vs observed defaults and losses by vintage |

## Run locally

```bash
cd dashboard
cp .env.example .env
docker compose up --build        # dashboard http://localhost:8050 · API docs http://localhost:8000/docs
```

Rebuild the model and data from the raw quarterly files (see `data/README.md` for the download):

```bash
python scripts/train_pd_model.py --src notebooks/data_splits --scored-dst build/scored
python scripts/prepare_data.py --src build/scored --dst dashboard/backend/app/data
pytest -q tests
```

## Limitations

- Lifetime PD is estimated per loan regardless of term; a production model would estimate 12-month and lifetime PD separately for IFRS 9.
- LGD and EAD are grade-level averages, not models.
- A single correlation ρ is applied to the whole book; Basel uses ρ between 0.03 and 0.16 for retail exposures, which the slider spans.
- `int_rate` and `sub_grade` are Lending Club's own risk pricing. They are legitimate origination inputs, but they make the model partly a recalibration of the lender's grades.
