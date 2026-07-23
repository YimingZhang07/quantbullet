from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantbullet.loan_simulation import (
    CashflowEngine,
    ConstantRecoveryLagProvider,
    ConstantSeverityProvider,
    ConstantTransitionModel,
    Loan,
    LoanSimulator,
    MatrixPaymentPolicy,
    PortfolioSimulator,
    StatusConfig,
    compute_period_metrics,
)


STATUSES = ["C", "D1M", "D2M", "D3M", "D4M", "PIF", "LIQ"]


def build_status_config() -> StatusConfig:
    return StatusConfig(
        valid_statuses=set(STATUSES),
        terminal_statuses={"PIF", "LIQ"},
        prepay_statuses={"PIF"},
        default_statuses={"LIQ"},
        delinquency_buckets={
            "D1M": "dq30_balance",
            "D2M": "dq60_balance",
            "D3M": "dq90_balance",
            "D4M": "dq120_balance",
        },
    )


def build_transition_table() -> dict[str, dict[str, float]]:
    return {
        "C": {"C": 0.62, "D1M": 0.10, "D2M": 0.00, "D3M": 0.00, "D4M": 0.00, "PIF": 0.18, "LIQ": 0.10},
        "D1M": {"C": 0.24, "D1M": 0.36, "D2M": 0.18, "D3M": 0.00, "D4M": 0.00, "PIF": 0.06, "LIQ": 0.16},
        "D2M": {"C": 0.12, "D1M": 0.18, "D2M": 0.34, "D3M": 0.14, "D4M": 0.00, "PIF": 0.04, "LIQ": 0.18},
        "D3M": {"C": 0.08, "D1M": 0.08, "D2M": 0.18, "D3M": 0.34, "D4M": 0.12, "PIF": 0.03, "LIQ": 0.17},
        "D4M": {"C": 0.05, "D1M": 0.05, "D2M": 0.05, "D3M": 0.15, "D4M": 0.38, "PIF": 0.02, "LIQ": 0.30},
        "PIF": {"C": 0.00, "D1M": 0.00, "D2M": 0.00, "D3M": 0.00, "D4M": 0.00, "PIF": 1.00, "LIQ": 0.00},
        "LIQ": {"C": 0.00, "D1M": 0.00, "D2M": 0.00, "D3M": 0.00, "D4M": 0.00, "PIF": 0.00, "LIQ": 1.00},
    }


def build_payment_matrix() -> dict[str, dict[str, int]]:
    return {
        "C": {"C": 1, "D1M": 0, "D2M": 0, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
        "D1M": {"C": 2, "D1M": 1, "D2M": 0, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
        "D2M": {"C": 3, "D1M": 2, "D2M": 1, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
        "D3M": {"C": 4, "D1M": 3, "D2M": 2, "D3M": 1, "D4M": 0, "PIF": 0, "LIQ": 0},
        "D4M": {"C": 5, "D1M": 4, "D2M": 3, "D3M": 2, "D4M": 1, "PIF": 0, "LIQ": 0},
        "PIF": {"C": 0, "D1M": 0, "D2M": 0, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
        "LIQ": {"C": 0, "D1M": 0, "D2M": 0, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
    }


def build_synthetic_loans(n_loans: int, seed: int) -> list[Loan]:
    rng = np.random.default_rng(seed)
    terms = rng.choice([24, 60, 120], size=n_loans, p=[0.30, 0.45, 0.25])
    loans = []
    for i, term in enumerate(terms, start=1):
        age = int(rng.integers(0, min(24, term - 1) + 1))
        original_balance = float(rng.uniform(5_000, 100_000))
        seasoning_factor = max((term - age) / term, 0.05)
        balance = original_balance * seasoning_factor * float(rng.uniform(0.90, 1.05))
        rate = float(rng.uniform(0.06, 0.12))
        loans.append(
            Loan(
                loan_id=f"L{i:04d}",
                balance=round(balance, 2),
                annual_rate=round(rate, 5),
                term_months=int(term),
                original_balance=round(original_balance, 2),
                age_months=age,
                status="C",
                metadata={"term_bucket": str(term)},
            )
        )
    return loans


def transition_table_frame(transition_table: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(transition_table, orient="index")[STATUSES].reset_index(names="from_status")


def payment_matrix_frame(payment_matrix: dict[str, dict[str, int]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(payment_matrix, orient="index")[STATUSES].reset_index(names="from_status")


def loans_frame(loans: list[Loan]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "loan_id": loan.loan_id,
                "balance": loan.balance,
                "original_balance": loan.original_balance,
                "annual_rate": loan.annual_rate,
                "term_months": loan.term_months,
                "age_months": loan.age_months,
                "remaining_term_months": loan.remaining_term_months,
                "scheduled_monthly_payment": loan.scheduled_monthly_payment,
                "status": loan.status,
                **dict(loan.metadata),
            }
            for loan in loans
        ]
    )


def write_excel(
    output_path: Path,
    *,
    run_config: pd.DataFrame,
    loans: pd.DataFrame,
    transition_table: pd.DataFrame,
    payment_matrix: pd.DataFrame,
    portfolio_cashflows: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path) as writer:
        run_config.to_excel(writer, sheet_name="run_config", index=False)
        loans.to_excel(writer, sheet_name="synthetic_loans", index=False)
        transition_table.to_excel(writer, sheet_name="transition_table", index=False)
        payment_matrix.to_excel(writer, sheet_name="payment_matrix", index=False)
        portfolio_cashflows.to_excel(writer, sheet_name="our_cashflows", index=False)
        portfolio_metrics.to_excel(writer, sheet_name="our_metrics", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic loan simulation reconciliation case.")
    parser.add_argument("--n-loans", type=int, default=50)
    parser.add_argument("--n-paths", type=int, default=250)
    parser.add_argument("--seed", type=int, default=202607)
    parser.add_argument("--start-date", default="2026-07")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".xlsx"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status_config = build_status_config()
    transition_table = build_transition_table()
    payment_matrix = build_payment_matrix()
    loans = build_synthetic_loans(args.n_loans, args.seed)
    horizon = max(loan.remaining_term_months for loan in loans)

    cashflow_engine = CashflowEngine(
        payment_policy=MatrixPaymentPolicy(payment_matrix, status_config=status_config),
        severity_provider=ConstantSeverityProvider(1.0),
        recovery_lag_provider=ConstantRecoveryLagProvider(0),
        status_config=status_config,
    )
    loan_simulator = LoanSimulator(
        transition_model=ConstantTransitionModel(transition_table, status_config=status_config),
        cashflow_engine=cashflow_engine,
        horizon=horizon,
        n_paths=args.n_paths,
        seed=args.seed,
        start_date=args.start_date,
    )

    start = time.perf_counter()
    result = PortfolioSimulator(loan_simulator).simulate(loans)
    portfolio_cashflows = result.portfolio_cashflows()
    portfolio_metrics = compute_period_metrics(portfolio_cashflows)
    elapsed = time.perf_counter() - start

    run_config = pd.DataFrame(
        [
            {"key": "n_loans", "value": args.n_loans},
            {"key": "n_paths", "value": args.n_paths},
            {"key": "horizon", "value": horizon},
            {"key": "seed", "value": args.seed},
            {"key": "start_date", "value": args.start_date},
            {"key": "severity", "value": 1.0},
            {"key": "recovery_lag", "value": 0},
            {"key": "elapsed_seconds", "value": round(elapsed, 3)},
        ]
    )

    write_excel(
        args.output,
        run_config=run_config,
        loans=loans_frame(loans),
        transition_table=transition_table_frame(transition_table),
        payment_matrix=payment_matrix_frame(payment_matrix),
        portfolio_cashflows=portfolio_cashflows,
        portfolio_metrics=portfolio_metrics,
    )
    print(
        f"Simulated {args.n_loans} loans x {args.n_paths} paths "
        f"x horizon {horizon} in {elapsed:.3f}s"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
