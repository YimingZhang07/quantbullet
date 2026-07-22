import pytest

from quantbullet.loan_simulation import (
    CashflowEngine,
    ConstantRecoveryLagProvider,
    ConstantSeverityProvider,
    Loan,
    MatrixPaymentPolicy,
    StatusConfig,
)


def _status_config():
    return StatusConfig(
        valid_statuses={"C", "D1M", "PIF", "LIQ"},
        terminal_statuses={"PIF", "LIQ"},
        prepay_statuses={"PIF"},
        default_statuses={"LIQ"},
        delinquency_buckets={"D1M": "dq30_balance"},
    )


def _payment_matrix():
    return {
        "C": {"C": 1, "D1M": 0, "PIF": 0, "LIQ": 0},
        "D1M": {"C": 2, "D1M": 1, "PIF": 0, "LIQ": 0},
        "PIF": {"C": 0, "D1M": 0, "PIF": 0, "LIQ": 0},
        "LIQ": {"C": 0, "D1M": 0, "PIF": 0, "LIQ": 0},
    }


def _cashflow_engine(severity=0.40, recovery_lag=3):
    config = _status_config()
    return CashflowEngine(
        payment_policy=MatrixPaymentPolicy(_payment_matrix(), status_config=config),
        severity_provider=ConstantSeverityProvider(severity),
        recovery_lag_provider=ConstantRecoveryLagProvider(recovery_lag),
        status_config=config,
    )


def test_scheduled_payment_baseline_uses_remaining_term_and_allows_override():
    seasoned = Loan("L1", balance=800.0, annual_rate=0.12, term_months=36, age_months=12)
    override = Loan(
        "L2",
        balance=800.0,
        annual_rate=0.12,
        term_months=36,
        age_months=12,
        scheduled_payment=50.0,
    )

    assert seasoned.scheduled_monthly_payment == pytest.approx(37.66, abs=0.01)
    assert override.scheduled_monthly_payment == 50.0


def test_matrix_policy_collects_multiple_scheduled_payments_sequentially():
    engine = _cashflow_engine()
    loan = Loan("L1", balance=1200.0, annual_rate=0.12, term_months=12, status="D1M")

    result = engine.project_period(loan, loan.initial_state(), "C")

    assert result.cashflow.scheduled_interest == pytest.approx(23.05, abs=0.01)
    assert result.cashflow.scheduled_principal == pytest.approx(190.18, abs=0.01)
    assert result.cashflow.end_balance == pytest.approx(1009.82, abs=0.01)
    assert result.next_state.status == "C"


def test_prepay_collects_current_interest_and_full_principal():
    engine = _cashflow_engine()
    loan = Loan("L1", balance=1200.0, annual_rate=0.12, term_months=12, status="C")

    result = engine.project_period(loan, loan.initial_state(), "PIF")

    assert result.cashflow.interest_collected == pytest.approx(12.0)
    assert result.cashflow.principal_collected == pytest.approx(1200.0)
    assert result.cashflow.end_balance == 0.0
    assert result.cashflow.prepayment_amount == pytest.approx(1105.38, abs=0.01)
    assert result.next_state.status == "PIF"


def test_default_recognizes_loss_and_schedules_recovery_event():
    engine = _cashflow_engine(severity=0.40, recovery_lag=3)
    loan = Loan("L1", balance=1200.0, annual_rate=0.12, term_months=12, status="C")

    result = engine.project_period(loan, loan.initial_state(), "LIQ", path_id=7)

    assert result.cashflow.default_balance == 1200.0
    assert result.cashflow.loss == 480.0
    assert result.cashflow.net_recovery == 0.0
    assert result.cashflow.end_balance == 0.0
    assert result.recovery_event is not None
    assert result.recovery_event.path_id == 7
    assert result.recovery_event.period == 3
    assert result.recovery_event.net_recovery == 720.0
