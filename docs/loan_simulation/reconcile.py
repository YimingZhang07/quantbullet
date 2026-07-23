"""Reconcile this framework's simulator against the roll-rate-model core.

用同一批 synthetic loans、同一套 transition table / payment matrix / seed / severity /
recovery timing，同时跑本 framework 的 simulator 和 roll-rate-model 的 Python
`run_cf_one`，再把 cashflows 和 metrics 写进一个对比 workbook。

Notes:
- roll-rate 用 constant `transition_fn` 注入，绕过 GAM model 依赖。
- payment matrix 会转置成 roll-rate 的 `pmt_matrix[to][from]` 方向。
- `our_rr_style_cpr` 是 diagnostic：用 roll-rate 的 PIF-balance CPR 口径重算本
  framework 的 paths，用来区分 "engine 差异" 和 "CPR 口径差异"。

Run (from repo root):
    $env:PYTHONPATH = "src"
    python docs/loan_simulation/reconcile.py --roll-rate-root C:/path/to/roll-rate-model

也可以用环境变量 ROLL_RATE_MODEL_ROOT 指定 roll-rate-model repo。
输出的 .xlsx 会写到本脚本同目录，并被该目录的 .gitignore 忽略。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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


def build_tieout_loans(n_loans: int, seed: int) -> list[Loan]:
    rng = np.random.default_rng(seed)
    terms = rng.choice([24, 60, 120], size=n_loans, p=[0.30, 0.45, 0.25])
    loans = []
    for i, term in enumerate(terms, start=1):
        balance = float(rng.uniform(5_000, 100_000))
        rate = float(rng.uniform(0.06, 0.12))
        loans.append(
            Loan(
                loan_id=f"L{i:04d}",
                balance=round(balance, 2),
                annual_rate=round(rate, 5),
                term_months=int(term),
                original_balance=round(balance, 2),
                age_months=0,
                scheduled_payment=_level_payment(balance, rate, int(term)),
                status="C",
                metadata={"term_bucket": str(term)},
            )
        )
    return loans


def run_our_simulation(
    loans: list[Loan],
    *,
    n_paths: int,
    horizon: int,
    seed: int,
    start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    status_config = build_status_config()
    cashflow_engine = CashflowEngine(
        payment_policy=MatrixPaymentPolicy(
            build_payment_matrix(),
            status_config=status_config,
        ),
        severity_provider=ConstantSeverityProvider(1.0),
        recovery_lag_provider=ConstantRecoveryLagProvider(0),
        status_config=status_config,
    )
    loan_simulator = LoanSimulator(
        transition_model=ConstantTransitionModel(
            build_transition_table(),
            status_config=status_config,
        ),
        cashflow_engine=cashflow_engine,
        horizon=horizon,
        n_paths=n_paths,
        seed=seed,
        start_date=start_date,
    )
    result = PortfolioSimulator(loan_simulator).simulate(loans)
    path_cashflows = result.path_cashflows()
    portfolio_cashflows = result.portfolio_cashflows()
    metrics = compute_period_metrics(portfolio_cashflows)
    return path_cashflows, portfolio_cashflows, metrics


def run_rollrate_simulation(
    loans: list[Loan],
    *,
    n_paths: int,
    horizon: int,
    seed: int,
    roll_rate_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _add_rollrate_python_path(roll_rate_root)
    from simengine.runner import CF_COL, compute_metrics, run_cf_one  # noqa: PLC0415

    transition_table = build_transition_table()
    dm = _rollrate_data_manager(horizon)
    portfolio_cf = [[0.0] * len(CF_COL) for _ in range(horizon)]

    for loan in loans:
        rollrate_loan = _to_rollrate_loan(loan)
        loan_cf = [[0.0] * len(CF_COL) for _ in range(horizon)]
        for path_id in range(n_paths):
            path_cf, _ = run_cf_one(
                rollrate_loan,
                dm,
                dup=1,
                seed=_stable_seed(seed, loan.loan_id, path_id),
                transition_fn=_constant_transition_fn(transition_table),
            )
            for period in range(horizon):
                for col_idx in range(len(CF_COL)):
                    loan_cf[period][col_idx] += path_cf[period][col_idx] / n_paths

        for period in range(horizon):
            for col_idx in range(len(CF_COL)):
                portfolio_cf[period][col_idx] += loan_cf[period][col_idx]

    cashflows = pd.DataFrame(portfolio_cf, columns=CF_COL)
    cashflows.insert(0, "period", range(1, horizon + 1))
    total_original_balance = sum(float(loan.original_balance) for loan in loans)
    metrics = pd.DataFrame(compute_metrics(portfolio_cf, total_original_balance))
    return cashflows, metrics


def rollrate_style_cpr_from_our_paths(
    path_cashflows: pd.DataFrame,
    loans: list[Loan],
) -> pd.DataFrame:
    """Compute CPR from our path rows using roll-rate's PIF-balance convention.

    This is a reconciliation diagnostic only. The framework's official CPR uses
    unscheduled principal. Roll-rate uses PIF event balance as the numerator and
    one-period scheduled principal in the denominator, so we reconstruct that
    convention here to isolate metric-definition differences from engine
    differences.
    """
    loan_terms = pd.DataFrame(
        [
            {
                "loan_id": loan.loan_id,
                "monthly_rate": loan.monthly_rate,
                "scheduled_payment": loan.scheduled_monthly_payment,
            }
            for loan in loans
        ]
    )
    frame = path_cashflows.merge(loan_terms, on="loan_id", how="left")
    frame["rollrate_scheduled_principal"] = (
        frame["scheduled_payment"] - frame["begin_balance"] * frame["monthly_rate"]
    ).clip(lower=0.0)
    frame["rollrate_scheduled_principal"] = np.minimum(
        frame["rollrate_scheduled_principal"],
        frame["begin_balance"],
    )
    frame["payoff_event_balance"] = np.where(
        frame["end_status"].eq("PIF"),
        frame["begin_balance"],
        0.0,
    )

    grouped = (
        frame.groupby("period", as_index=False)[
            ["begin_balance", "rollrate_scheduled_principal", "payoff_event_balance"]
        ]
        .sum()
        .sort_values("period")
        .reset_index(drop=True)
    )
    denominator = (
        grouped["begin_balance"] - grouped["rollrate_scheduled_principal"]
    ).clip(lower=0.0)
    smm = (grouped["payoff_event_balance"] / denominator.replace(0, float("nan"))).fillna(0.0)
    grouped["our_rollrate_style_cpr"] = 1.0 - (1.0 - smm.clip(upper=1.0)) ** 12
    return grouped[["period", "our_rollrate_style_cpr"]]


def metrics_diff(
    our_metrics: pd.DataFrame,
    rollrate_metrics: pd.DataFrame,
    our_rollrate_style_metrics: pd.DataFrame,
) -> pd.DataFrame:
    our = our_metrics[["period", "cpr", "cdr", "cumulative_loss_rate"]].rename(
        columns={
            "cpr": "our_cpr",
            "cdr": "our_cdr",
            "cumulative_loss_rate": "our_cgl",
        }
    )
    rollrate = rollrate_metrics[["period", "cpr", "cdr", "cgl"]].rename(
        columns={
            "cpr": "rollrate_cpr",
            "cdr": "rollrate_cdr",
            "cgl": "rollrate_cgl",
        }
    )
    comparison = (
        our.merge(rollrate, on="period", how="outer")
        .merge(our_rollrate_style_metrics, on="period", how="outer")
        .sort_values("period")
        .reset_index(drop=True)
    )
    comparison["cpr_diff"] = comparison["our_cpr"] - comparison["rollrate_cpr"]
    comparison["rollrate_style_cpr_diff"] = (
        comparison["our_rollrate_style_cpr"] - comparison["rollrate_cpr"]
    )
    comparison["cdr_diff"] = comparison["our_cdr"] - comparison["rollrate_cdr"]
    comparison["cgl_diff"] = comparison["our_cgl"] - comparison["rollrate_cgl"]
    return comparison


def cashflow_diff(our_cashflows: pd.DataFrame, rollrate_cashflows: pd.DataFrame) -> pd.DataFrame:
    our = our_cashflows[
        ["period", "begin_balance", "end_balance", "interest_collected", "principal_collected", "loss"]
    ].rename(
        columns={
            "begin_balance": "our_begin_balance",
            "end_balance": "our_end_balance",
            "interest_collected": "our_interest",
            "principal_collected": "our_principal",
            "loss": "our_loss",
        }
    )
    rollrate = rollrate_cashflows[
        ["period", "begin_bal", "end_bal", "int_pmt", "prin_pmt", "loss"]
    ].rename(
        columns={
            "begin_bal": "rollrate_begin_balance",
            "end_bal": "rollrate_end_balance",
            "int_pmt": "rollrate_interest",
            "prin_pmt": "rollrate_principal",
            "loss": "rollrate_loss",
        }
    )
    comparison = (
        our.merge(rollrate, on="period", how="outer")
        .sort_values("period")
        .reset_index(drop=True)
    )
    for field in ["begin_balance", "end_balance", "interest", "principal", "loss"]:
        comparison[f"{field}_diff"] = (
            comparison[f"our_{field}"] - comparison[f"rollrate_{field}"]
        )
    return comparison


def write_excel(
    output_path: Path,
    *,
    run_config: pd.DataFrame,
    loans: pd.DataFrame,
    transition_table: pd.DataFrame,
    payment_matrix: pd.DataFrame,
    our_cashflows: pd.DataFrame,
    rollrate_cashflows: pd.DataFrame,
    our_metrics: pd.DataFrame,
    our_rollrate_style_metrics: pd.DataFrame,
    rollrate_metrics: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path) as writer:
        run_config.to_excel(writer, sheet_name="run_config", index=False)
        loans.to_excel(writer, sheet_name="synthetic_loans", index=False)
        transition_table.to_excel(writer, sheet_name="transition_table", index=False)
        payment_matrix.to_excel(writer, sheet_name="payment_matrix", index=False)
        our_cashflows.to_excel(writer, sheet_name="our_cashflows", index=False)
        rollrate_cashflows.to_excel(writer, sheet_name="rollrate_cashflows", index=False)
        our_metrics.to_excel(writer, sheet_name="our_metrics", index=False)
        our_rollrate_style_metrics.to_excel(
            writer,
            sheet_name="our_rr_style_cpr",
            index=False,
        )
        rollrate_metrics.to_excel(writer, sheet_name="rollrate_metrics", index=False)
        metrics_diff(our_metrics, rollrate_metrics, our_rollrate_style_metrics).to_excel(
            writer,
            sheet_name="metrics_diff",
            index=False,
        )
        cashflow_diff(our_cashflows, rollrate_cashflows).to_excel(
            writer,
            sheet_name="cashflow_diff",
            index=False,
        )


def _rollrate_data_manager(horizon: int):
    payment_matrix = build_payment_matrix()
    pmt_matrix_to_from = {
        to_status: {
            from_status: payment_matrix[from_status][to_status]
            for from_status in STATUSES
        }
        for to_status in STATUSES
    }
    return SimpleNamespace(
        n_per=horizon,
        models={},
        _transition_layout={},
        terminal_statuses={"PIF", "LIQ"},
        status_to_roll={status: list(build_transition_table()[status]) for status in STATUSES},
        pmt_matrix=pmt_matrix_to_from,
        liq_severity=1.0,
        dq_buckets={
            "D1M": ("dq30", "dq30_bal"),
            "D2M": ("dq60", "dq60_bal"),
            "D3M": ("dq90", "dq90_bal"),
            "D4M": ("dq120", "dq120_bal"),
        },
    )


def _constant_transition_fn(transition_table: dict[str, dict[str, float]]):
    def transition_fn(loan, from_status, roll_to, dm, per, rng, **kwargs):
        probabilities = [transition_table[from_status][to_status] for to_status in roll_to]
        draw = rng.random()
        cumulative = 0.0
        for to_status, probability in zip(roll_to, probabilities):
            cumulative += probability
            if draw <= cumulative:
                return to_status, probabilities
        return roll_to[-1], probabilities

    return transition_fn


def _to_rollrate_loan(loan: Loan) -> dict:
    return {
        "loan_id": loan.loan_id,
        "status": loan.status,
        "end_bal": loan.balance,
        "orig_bal": loan.original_balance,
        "term": loan.term_months,
        "int_rate": loan.annual_rate,
        "loan_age": loan.age_months,
        "r_dt": "2026-07-31",
    }


def _level_payment(balance: float, annual_rate: float, term_months: int) -> float:
    monthly_rate = annual_rate / 12.0
    if monthly_rate == 0:
        return balance / term_months
    discount_factor = (1 + monthly_rate) ** term_months
    return balance * monthly_rate * discount_factor / (discount_factor - 1)


def _stable_seed(seed: int, loan_id: str, path_id: int) -> int:
    value = f"{seed}|{loan_id}|{path_id}"
    hash_value = 1469598103934665603
    for character in value:
        hash_value ^= ord(character)
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(hash_value % (2**32))


def _add_rollrate_python_path(roll_rate_root: Path) -> None:
    python_dir = roll_rate_root / "python"
    if not python_dir.exists():
        raise FileNotFoundError(f"roll-rate-model python directory not found: {python_dir}")
    sys.path.insert(0, str(python_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare quantbullet simulator to roll-rate Python core.")
    parser.add_argument("--n-loans", type=int, default=50)
    parser.add_argument("--n-paths", type=int, default=250)
    parser.add_argument("--seed", type=int, default=202607)
    parser.add_argument("--start-date", default="2026-07")
    parser.add_argument(
        "--roll-rate-root",
        type=Path,
        default=None,
        help="Path to the roll-rate-model repo. Can also use ROLL_RATE_MODEL_ROOT.",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".xlsx"))
    return parser.parse_args()


def _resolve_roll_rate_root(roll_rate_root: Path | None) -> Path:
    if roll_rate_root is not None:
        return roll_rate_root

    env_value = os.environ.get("ROLL_RATE_MODEL_ROOT")
    if env_value:
        return Path(env_value)

    raise SystemExit(
        "roll-rate-model repo path is required. Provide --roll-rate-root "
        "or set ROLL_RATE_MODEL_ROOT."
    )


def main() -> None:
    args = parse_args()
    roll_rate_root = _resolve_roll_rate_root(args.roll_rate_root)
    loans = build_tieout_loans(args.n_loans, args.seed)
    horizon = max(loan.term_months for loan in loans)

    start = time.perf_counter()
    our_path_cashflows, our_cashflows, our_metrics = run_our_simulation(
        loans,
        n_paths=args.n_paths,
        horizon=horizon,
        seed=args.seed,
        start_date=args.start_date,
    )
    rollrate_cashflows, rollrate_metrics = run_rollrate_simulation(
        loans,
        n_paths=args.n_paths,
        horizon=horizon,
        seed=args.seed,
        roll_rate_root=roll_rate_root,
    )
    our_rollrate_style_metrics = rollrate_style_cpr_from_our_paths(
        our_path_cashflows,
        loans,
    )
    elapsed = time.perf_counter() - start

    run_config = pd.DataFrame(
        [
            {"key": "n_loans", "value": args.n_loans},
            {"key": "n_paths", "value": args.n_paths},
            {"key": "horizon", "value": horizon},
            {"key": "seed", "value": args.seed},
            {"key": "severity", "value": 1.0},
            {"key": "recovery_lag", "value": 0},
            {"key": "elapsed_seconds", "value": round(elapsed, 3)},
            {"key": "roll_rate_root", "value": str(roll_rate_root)},
        ]
    )

    write_excel(
        args.output,
        run_config=run_config,
        loans=loans_frame(loans),
        transition_table=transition_table_frame(build_transition_table()),
        payment_matrix=payment_matrix_frame(build_payment_matrix()),
        our_cashflows=our_cashflows,
        rollrate_cashflows=rollrate_cashflows,
        our_metrics=our_metrics,
        our_rollrate_style_metrics=our_rollrate_style_metrics,
        rollrate_metrics=rollrate_metrics,
    )
    print(
        f"Compared {args.n_loans} loans x {args.n_paths} paths "
        f"x horizon {horizon} in {elapsed:.3f}s"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
