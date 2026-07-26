"""Run PAR_2025_1 + GENERIC_v4 with roll-rate-model only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd


EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = EXAMPLE_DIR / "rollrate_cashflows.xlsx"

DEAL_NAME = "PAR_2025_1"
COEFFICIENT_SET = "GENERIC_v4"
LOANS_PREPPED_PATH = "input/deals/PAR_2025_1/loans_prepped.json"
FROM_STATUSES = ["C", "D1M", "D2M", "D3M", "D4M"]
HORIZON = 12
N_PATHS = 100
SEED = 20260725

STATUS_TO_ROLL = {
    "C": ["C", "D1M", "D2M", "D3M", "D4M", "PIF", "LIQ"],
    "D1M": ["D1M", "C", "D2M", "D3M", "D4M", "PIF", "LIQ"],
    "D2M": ["D2M", "C", "D1M", "D3M", "D4M", "PIF", "LIQ"],
    "D3M": ["D3M", "C", "D1M", "D2M", "D4M", "PIF", "LIQ"],
    "D4M": ["D4M", "C", "D1M", "D2M", "D3M", "PIF", "LIQ"],
}
TERMINAL_STATUSES = {"PIF", "LIQ"}
DELINQUENCY_BUCKETS = {
    "D1M": ("dq30", "dq30_bal"),
    "D2M": ("dq60", "dq60_bal"),
    "D3M": ("dq90", "dq90_bal"),
    "D4M": ("dq120", "dq120_bal"),
}


def load_prepped_loans(roll_rate_root: Path) -> list[dict[str, Any]]:
    with (roll_rate_root / LOANS_PREPPED_PATH).open("r", encoding="utf-8") as file:
        loans = json.load(file)
    if not isinstance(loans, list):
        raise ValueError(f"Expected a list of loans in {LOANS_PREPPED_PATH}")
    return loans


def load_payment_matrix(roll_rate_root: Path) -> dict[str, dict[str, int]]:
    frame = pd.read_csv(roll_rate_root / "input" / "pmt_matrix.txt", sep="\t", index_col=0)
    statuses = {status for row in STATUS_TO_ROLL.values() for status in row}
    matrix = {}
    for begin_status in statuses:
        matrix[begin_status] = {}
        for end_status in statuses:
            value = frame.loc[begin_status, end_status] if begin_status in frame.index else 0
            matrix[begin_status][end_status] = 0 if end_status in TERMINAL_STATUSES else int(value)
    return matrix


def build_data_manager(roll_rate_root: Path, horizon: int):
    _add_rollrate_python_path(roll_rate_root)
    from simengine.data_prep import (  # noqa: PLC0415
        _get_registry,
        build_all_models,
        classify_model_terms,
        read_coef_file,
    )
    from simengine.runner import _build_transition_layout  # noqa: PLC0415

    coefficient_root = roll_rate_root / "input" / "coef" / COEFFICIENT_SET
    coef_by_from = {
        from_status: read_coef_file(str(coefficient_root / f"from{from_status}.txt"))
        for from_status in FROM_STATUSES
    }
    models = build_all_models(coef_by_from)
    classify_model_terms(models, _get_registry().time_varying_names())

    payment_matrix = load_payment_matrix(roll_rate_root)
    pmt_matrix_to_from = {
        to_status: {
            from_status: payment_matrix[from_status][to_status]
            for from_status in payment_matrix
        }
        for to_status in {status for row in STATUS_TO_ROLL.values() for status in row}
    }
    dm = SimpleNamespace(
        n_per=horizon,
        models=models,
        status_to_roll=STATUS_TO_ROLL,
        terminal_statuses=TERMINAL_STATUSES,
        pmt_matrix=pmt_matrix_to_from,
        liq_severity=1.0,
        dq_buckets=DELINQUENCY_BUCKETS,
        clean_status_dict={
            status: status.split(".")[0]
            for statuses in STATUS_TO_ROLL.values()
            for status in statuses
        },
        prob_layout={},
        dial_data={},
    )
    dm._transition_layout = _build_transition_layout(dm)
    return dm


def run_rollrate(
    loans: list[dict[str, Any]],
    *,
    dm,
    horizon: int,
    n_paths: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from simengine.runner import CF_COL, compute_metrics, run_cf_one  # noqa: PLC0415

    eligible_loans = [
        dict(loan)
        for loan in loans
        if float(loan.get("end_bal", 0.0)) > 0.1
    ]
    portfolio_cf = [[0.0] * len(CF_COL) for _ in range(horizon)]
    for loan in eligible_loans:
        loan_cf = [[0.0] * len(CF_COL) for _ in range(horizon)]
        for path_id in range(n_paths):
            path_cf, _ = run_cf_one(
                loan,
                dm,
                dup=1,
                seed=_stable_seed(seed, str(loan["loan_id"]), path_id),
            )
            for period in range(horizon):
                for col_idx in range(len(CF_COL)):
                    loan_cf[period][col_idx] += path_cf[period][col_idx] / n_paths
        for period in range(horizon):
            for col_idx in range(len(CF_COL)):
                portfolio_cf[period][col_idx] += loan_cf[period][col_idx]

    cashflows = pd.DataFrame(portfolio_cf, columns=CF_COL)
    cashflows.insert(0, "period", range(1, horizon + 1))
    total_orig_bal = sum(float(loan["orig_bal"]) for loan in eligible_loans)
    metrics = pd.DataFrame(compute_metrics(portfolio_cf, total_orig_bal))
    return cashflows, metrics


def _stable_seed(seed: int, loan_id: str, path_id: int) -> int:
    value = f"{seed}|{loan_id}|{path_id}"
    hash_value = 1469598103934665603
    for character in value:
        hash_value ^= ord(character)
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(hash_value % (2**32))


def write_excel(
    output_path: Path,
    *,
    cashflows: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    run_config = pd.DataFrame(
        [
            {"key": "deal_name", "value": DEAL_NAME},
            {"key": "coefficient_set", "value": COEFFICIENT_SET},
            {"key": "horizon", "value": HORIZON},
            {"key": "n_paths", "value": N_PATHS},
            {"key": "seed", "value": SEED},
        ]
    )
    with pd.ExcelWriter(output_path) as writer:
        run_config.to_excel(writer, sheet_name="run_config", index=False)
        metrics.to_excel(writer, sheet_name="metrics", index=False)
        cashflows.to_excel(writer, sheet_name="cashflows", index=False)


def _add_rollrate_python_path(roll_rate_root: Path) -> None:
    python_dir = roll_rate_root / "python"
    if not python_dir.exists():
        raise FileNotFoundError(f"roll-rate-model python directory not found: {python_dir}")
    sys.path.insert(0, str(python_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the roll-rate reference demo.")
    parser.add_argument(
        "--roll-rate-root",
        type=Path,
        default=None,
        help="Path to roll-rate-model. Can also use ROLL_RATE_MODEL_ROOT.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
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
    loans = load_prepped_loans(roll_rate_root)[:20]
    dm = build_data_manager(roll_rate_root, HORIZON)
    cashflows, metrics = run_rollrate(
        loans,
        dm=dm,
        horizon=HORIZON,
        n_paths=N_PATHS,
        seed=SEED,
    )
    write_excel(args.output, cashflows=cashflows, metrics=metrics)
    print(metrics[["period", "cpr", "cdr", "cgl"]].to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
