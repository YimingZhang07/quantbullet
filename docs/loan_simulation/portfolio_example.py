"""Loan simulation portfolio example.

生成一个 5-loan 的 demo portfolio，跑 seeded Monte Carlo simulation，然后把
path-level、loan-level、portfolio-level cashflows 和 portfolio metrics 写进一个
Excel workbook。主要用来直观检查 framework 的端到端输出。

Run (from repo root):
    $env:PYTHONPATH = "src"
    python docs/loan_simulation/portfolio_example.py

输出的 .xlsx 会写到本脚本同目录，并被该目录的 .gitignore 忽略。
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from quantbullet.loan_simulation import (
    CashflowEngine,
    ConstantRecoveryLagProvider,
    ConstantSeverityProvider,
    ConstantTransitionModel,
    DataFrameMacroFeatureProvider,
    Loan,
    LoanSimulator,
    MatrixPaymentPolicy,
    PortfolioSimulator,
    StatusConfig,
    compute_period_metrics,
)


def build_status_config() -> StatusConfig:
    return StatusConfig(
        valid_statuses={"C", "D1M", "D2M", "PIF", "LIQ"},
        terminal_statuses={"PIF", "LIQ"},
        prepay_statuses={"PIF"},
        default_statuses={"LIQ"},
        delinquency_buckets={
            "D1M": "dq30_balance",
            "D2M": "dq60_balance",
        },
    )


def build_transition_model(status_config: StatusConfig) -> ConstantTransitionModel:
    # Deliberately high transition probabilities make the example easier to inspect.
    transitions = {
        "C": {"C": 0.55, "D1M": 0.20, "D2M": 0.00, "PIF": 0.15, "LIQ": 0.10},
        "D1M": {"C": 0.25, "D1M": 0.35, "D2M": 0.20, "PIF": 0.05, "LIQ": 0.15},
        "D2M": {"C": 0.15, "D1M": 0.15, "D2M": 0.35, "PIF": 0.05, "LIQ": 0.30},
        "PIF": {"C": 0.00, "D1M": 0.00, "D2M": 0.00, "PIF": 1.00, "LIQ": 0.00},
        "LIQ": {"C": 0.00, "D1M": 0.00, "D2M": 0.00, "PIF": 0.00, "LIQ": 1.00},
    }
    return ConstantTransitionModel(transitions, status_config=status_config)


def build_payment_policy(status_config: StatusConfig) -> MatrixPaymentPolicy:
    return MatrixPaymentPolicy.from_delinquency_chain(
        ["C", "D1M", "D2M"],
        status_config=status_config,
    )


def build_macro_provider(start_date: str, periods: int) -> DataFrameMacroFeatureProvider:
    dates = pd.period_range(start=start_date, periods=periods + 1, freq="M")[1:]
    macro = pd.DataFrame(
        {
            "hpi": [100 + 0.25 * i for i in range(len(dates))],
            "market_rate": [0.065 + 0.0005 * i for i in range(len(dates))],
        },
        index=dates.to_timestamp(how="end"),
    )
    return DataFrameMacroFeatureProvider(macro)


def build_loans() -> list[Loan]:
    return [
        Loan("L1", balance=8_000, annual_rate=0.080, term_months=24, original_balance=9_500, age_months=2, status="C"),
        Loan("L2", balance=12_000, annual_rate=0.095, term_months=24, original_balance=14_000, age_months=6, status="C"),
        Loan("L3", balance=25_000, annual_rate=0.075, term_months=60, original_balance=30_000, age_months=10, status="C"),
        Loan("L4", balance=40_000, annual_rate=0.070, term_months=60, original_balance=50_000, age_months=18, status="C"),
        Loan("L5", balance=75_000, annual_rate=0.065, term_months=120, original_balance=100_000, age_months=24, status="C"),
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
        severity_provider=ConstantSeverityProvider(1.00),
        recovery_lag_provider=ConstantRecoveryLagProvider(0),
        status_config=status_config,
    )
    loan_simulator = LoanSimulator(
        transition_model=build_transition_model(status_config),
        cashflow_engine=cashflow_engine,
        horizon=horizon,
        n_paths=100,
        seed=202607,
        start_date=start_date,
        macro_provider=build_macro_provider(start_date, horizon),
    )

    start = time.perf_counter()
    result = PortfolioSimulator(loan_simulator).simulate(loans)

    path_cashflows = result.path_cashflows()
    loan_cashflows = result.loan_cashflows()
    portfolio_cashflows = result.portfolio_cashflows()
    portfolio_metrics = compute_period_metrics(portfolio_cashflows)

    output_path = Path(__file__).with_suffix(".xlsx")
    write_excel(
        output_path,
        path_cashflows=path_cashflows,
        loan_cashflows=loan_cashflows,
        portfolio_cashflows=portfolio_cashflows,
        portfolio_metrics=portfolio_metrics,
    )
    elapsed = time.perf_counter() - start
    print(f"Wrote {output_path}")
    print(
        f"Simulated {len(loans)} loans x {loan_simulator.n_paths} paths "
        f"x horizon {loan_simulator.horizon} in {elapsed:.3f}s"
    )


if __name__ == "__main__":
    main()
