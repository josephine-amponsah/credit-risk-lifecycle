import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard" / "backend"))
from app.services import risk_metrics as rm  # noqa: E402


@pytest.fixture
def book():
    rng = np.random.default_rng(0)
    n = 2000
    grade = rng.choice(list("ABC"), n)
    pd_true = np.select([grade == "A", grade == "B"], [0.05, 0.12], 0.25)
    default = rng.random(n) < pd_true
    funded = rng.uniform(5_000, 30_000, n)
    repaid = np.where(default, funded * rng.uniform(0.2, 0.6, n), funded)
    return pd.DataFrame({
        "issue_d": pd.Timestamp("2015-01-01"), "grade": grade, "funded_amnt": funded,
        "total_rec_prncp": repaid, "recoveries": np.where(default, (funded - repaid) * 0.1, 0.0),
        "collection_recovery_fee": 0.0,
        "loan_status": np.where(default, "Charged Off", "Fully Paid"),
        "last_pymnt_d": "Jun-2016", "pd_calibrated": pd_true,
    })


def test_does_not_meet_policy_fully_paid_is_not_default():
    assert not rm.is_default_status("Does not meet the credit policy. Status:Fully Paid")
    assert rm.is_default_status("Does not meet the credit policy. Status:Charged Off")
    assert not rm.is_default_status("In Grace Period")


def test_paid_off_loans_are_censored_at_payoff():
    df = pd.DataFrame({"issue_d": ["Jan-2015"], "loan_status": ["Fully Paid"], "last_pymnt_d": ["Jan-2017"]})
    assert rm.months_since_issue(df).iloc[0] == pytest.approx(24, abs=0.5)


def test_lgd_is_estimated_from_recoveries(book):
    params = rm.lgd_ead_by_grade(book)
    assert params["lgd"].between(0.85, 0.95).all()  # 10% recovered


def test_expected_loss_close_to_realised(book):
    total_el, _, by_grade = rm.expected_loss(book)
    realised = rm.realised_loss(book).sum()
    assert total_el == pytest.approx(realised, rel=0.15)
    assert set(by_grade["grade"]) == {"A", "B", "C"}


def test_vasicek_var_matches_simulation_and_rises_with_correlation(book):
    sim = rm.var_from_simulations(rm.simulate_portfolio_losses(book, nsim=20_000, rho=0.1), 0.99)
    closed = rm.vasicek_var(book, confidence=0.99, rho=0.1)
    assert sim == pytest.approx(closed, rel=0.05)
    assert rm.vasicek_var(book, 0.99, rho=0.15) > rm.vasicek_var(book, 0.99, rho=0.03)
