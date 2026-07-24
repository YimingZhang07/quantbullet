"""Model-backed transition example for the loan simulation framework.

This example uses `CompositeTransitionModel` instead of a constant transition
table. A prepay edge is driven by macro rate incentive, while delinquency/cure
edges read path-level delinquency history maintained by `PathFeatureTracker`.

Run (from repo root):
    $env:PYTHONPATH = "src"
    python docs/loan_simulation/model_backed_example.py

The generated .xlsx is written next to this script and ignored by git.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from quantbullet.loan_simulation import (
    CashflowEngine,
    CompositeTransitionModel,
    ConstantRecoveryLagProvider,
    ConstantSeverityProvider,
    DataFrameMacroFeatureProvider,
    FeatureContext,
    Loan,
    LoanSimulator,
    MatrixPaymentPolicy,
    PortfolioSimulator,
    StatusConfig,
    compute_period_metrics,
)


def build_status_config() -> StatusConfig:
    return StatusConfig(
        valid_statuses={"C", "D1M", "PIF", "LIQ"},
        terminal_statuses={"PIF", "LIQ"},
        prepay_statuses={"PIF"},
        default_statuses={"LIQ"},
        delinquency_buckets={"D1M": "dq30_balance"},
    )


def prepay_probability(context: FeatureContext) -> float:
    """Toy prepay model: higher incentive and seasoning increase payoff odds."""
    market_rate = context.macro_features["market_rate"]
    incentive = max(context.loan.annual_rate - market_rate, 0.0)
    seasoning = min(context.current_state.age_months / 60.0, 1.0)
    probability = 0.03 + 2.5 * incentive + 0.04 * seasoning
    return max(0.0, min(probability, 0.35))


def delinquency_probability(context: FeatureContext) -> float:
    """Toy delinquency model: prior delinquency raises future delinquency odds."""
    probability = 0.03
    if context.path_features["ever_delinquent"]:
        probability += 0.05
    if context.current_state.balance / context.loan.original_balance > 0.80:
        probability += 0.02
    return min(probability, 0.20)


def cure_probability(context: FeatureContext) -> float:
    """Toy cure model: cure probability falls with consecutive delinquency."""
    consecutive_dq = context.path_features["consecutive_delinquent_months"]
    return max(0.15, 0.45 - 0.10 * consecutive_dq)


def build_transition_model(status_config: StatusConfig) -> CompositeTransitionModel:
    return CompositeTransitionModel(
        edges={
            "C": {
                "PIF": prepay_probability,
                "D1M": delinquency_probability,
                "LIQ": 0.005,
            },
            "D1M": {
                "C": cure_probability,
                "LIQ": 0.04,
            },
        },
        status_config=status_config,
    )


def build_payment_policy(status_config: StatusConfig) -> MatrixPaymentPolicy:
    payment_periods = {
        "C": {"C": 1, "D1M": 0, "PIF": 0, "LIQ": 0},
        "D1M": {"C": 2, "D1M": 1, "PIF": 0, "LIQ": 0},
        "PIF": {"C": 0, "D1M": 0, "PIF": 0, "LIQ": 0},
        "LIQ": {"C": 0, "D1M": 0, "PIF": 0, "LIQ": 0},
    }
    return MatrixPaymentPolicy(payment_periods, status_config=status_config)


def build_macro_provider(start_date: str, horizon: int) -> DataFrameMacroFeatureProvider:
    dates = pd.period_range(start=start_date, periods=horizon + 1, freq="M")[1:]
    macro = pd.DataFrame(
        {
            "market_rate": [0.075 - 0.0004 * i for i in range(len(dates))],
            "hpi": [100 + 0.20 * i for i in range(len(dates))],
        },
        index=dates.to_timestamp(how="end"),
    )
    return DataFrameMacroFeatureProvider(macro)


def build_loans() -> list[Loan]:
    return [
        Loan("M1", 10_000, 0.090, 24, original_balance=11_000, age_months=4, status="C"),
        Loan("M2", 18_000, 0.105, 60, original_balance=22_000, age_months=12, status="C"),
        Loan("M3", 35_000, 0.080, 60, original_balance=40_000, age_months=18, status="C"),
        Loan("M4", 55_000, 0.072, 120, original_balance=75_000, age_months=30, status="C"),
        Loan("M5", 90_000, 0.068, 120, original_balance=120_000, age_months=36, status="C"),
    ]


def write_excel(
    output_path: Path,
    *,
    path_cashflows: pd.DataFrame,
    loan_cashflows: pd.DataFrame,
    portfolio_cashflows: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path) as writer:
        portfolio_cashflows.to_excel(writer, sheet_name="portfolio_cashflows", index=False)
        portfolio_metrics.to_excel(writer, sheet_name="portfolio_metrics", index=False)
        for loan_id, loan_frame in loan_cashflows.groupby("loan_id", sort=True):
            loan_frame.to_excel(writer, sheet_name=f"loan_{loan_id}", index=False)
        path_cashflows.to_excel(writer, sheet_name="path_cashflows", index=False)


def main() -> None:
    start_date = "2026-07"
    loans = build_loans()
    horizon = max(loan.remaining_term_months for loan in loans)
    status_config = build_status_config()
    cashflow_engine = CashflowEngine(
        payment_policy=build_payment_policy(status_config),
        severity_provider=ConstantSeverityProvider(0.50),
        recovery_lag_provider=ConstantRecoveryLagProvider(3),
        status_config=status_config,
    )
    loan_simulator = LoanSimulator(
        transition_model=build_transition_model(status_config),
        cashflow_engine=cashflow_engine,
        horizon=horizon,
        n_paths=200,
        seed=202607,
        start_date=start_date,
        macro_provider=build_macro_provider(start_date, horizon),
    )

    start = time.perf_counter()
    result = PortfolioSimulator(loan_simulator).simulate(loans)
    portfolio_cashflows = result.portfolio_cashflows()
    portfolio_metrics = compute_period_metrics(portfolio_cashflows)
    elapsed = time.perf_counter() - start

    output_path = Path(__file__).with_suffix(".xlsx")
    write_excel(
        output_path,
        path_cashflows=result.path_cashflows(),
        loan_cashflows=result.loan_cashflows(),
        portfolio_cashflows=portfolio_cashflows,
        portfolio_metrics=portfolio_metrics,
    )
    print(f"Wrote {output_path}")
    print(
        f"Simulated {len(loans)} loans x {loan_simulator.n_paths} paths "
        f"x horizon {loan_simulator.horizon} in {elapsed:.3f}s"
    )


if __name__ == "__main__":
    main()
