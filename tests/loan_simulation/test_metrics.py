import pandas as pd
import pytest

from quantbullet.loan_simulation import compute_period_metrics


def test_compute_period_metrics_calculates_prepay_default_and_loss_rates():
    cashflows = pd.DataFrame(
        [
            {
                "period": 1,
                "period_date": "2026-08",
                "original_balance": 120.0,
                "begin_balance": 100.0,
                "end_balance": 80.0,
                "scheduled_principal": 10.0,
                "principal_collected": 20.0,
                "prepayment_amount": 10.0,
                "default_balance": 0.0,
                "loss": 0.0,
                "net_recovery": 0.0,
                "delinquent_balance": 0.0,
            },
            {
                "period": 2,
                "period_date": "2026-09",
                "original_balance": 120.0,
                "begin_balance": 80.0,
                "end_balance": 60.0,
                "scheduled_principal": 5.0,
                "principal_collected": 5.0,
                "prepayment_amount": 0.0,
                "default_balance": 20.0,
                "loss": 8.0,
                "net_recovery": 2.0,
                "delinquent_balance": 10.0,
            },
        ]
    )

    metrics = compute_period_metrics(cashflows)

    assert metrics.loc[0, "smm"] == pytest.approx(10.0 / 90.0)
    assert metrics.loc[0, "cpr"] == pytest.approx(1.0 - (1.0 - 10.0 / 90.0) ** 12)
    assert metrics.loc[1, "mdr"] == pytest.approx(20.0 / 80.0)
    assert metrics.loc[1, "cdr"] == pytest.approx(1.0 - (1.0 - 20.0 / 80.0) ** 12)
    assert metrics.loc[1, "period_net_loss"] == 6.0
    assert metrics.loc[1, "cumulative_loss"] == 8.0
    assert metrics.loc[1, "cumulative_loss_rate"] == pytest.approx(8.0 / 120.0)
    assert metrics.loc[1, "cumulative_net_loss"] == 6.0
    assert metrics.loc[1, "delinquency_rate"] == pytest.approx(10.0 / 60.0)


def test_compute_period_metrics_supports_additional_grouping_columns():
    cashflows = pd.DataFrame(
        [
            {
                "segment": "A",
                "period": 1,
                "original_balance": 100.0,
                "begin_balance": 100.0,
                "end_balance": 90.0,
                "scheduled_principal": 10.0,
                "principal_collected": 10.0,
                "prepayment_amount": 0.0,
                "default_balance": 0.0,
                "loss": 1.0,
                "net_recovery": 0.0,
                "delinquent_balance": 0.0,
            },
            {
                "segment": "B",
                "period": 1,
                "original_balance": 200.0,
                "begin_balance": 200.0,
                "end_balance": 180.0,
                "scheduled_principal": 20.0,
                "principal_collected": 20.0,
                "prepayment_amount": 0.0,
                "default_balance": 0.0,
                "loss": 4.0,
                "net_recovery": 1.0,
                "delinquent_balance": 0.0,
            },
        ]
    )

    metrics = compute_period_metrics(cashflows, group_by=["segment"])

    assert metrics["segment"].tolist() == ["A", "B"]
    assert metrics["cumulative_loss"].tolist() == [1.0, 4.0]
    assert metrics["cumulative_net_loss"].tolist() == [1.0, 3.0]


def test_compute_period_metrics_requires_original_balance():
    cashflows = pd.DataFrame(
        [
            {
                "period": 1,
                "begin_balance": 100.0,
                "end_balance": 90.0,
                "scheduled_principal": 10.0,
                "principal_collected": 10.0,
                "prepayment_amount": 0.0,
                "default_balance": 0.0,
                "loss": 0.0,
                "net_recovery": 0.0,
                "delinquent_balance": 0.0,
            }
        ]
    )

    with pytest.raises(KeyError, match="original_balance"):
        compute_period_metrics(cashflows)
