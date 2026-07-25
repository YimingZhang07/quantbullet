"""Tie out roll-rate GAM coefficients against quantbullet replay.

This script compares roll-rate coefficient files at two levels:

1. raw edge logits from the parsed GAM terms; and
2. the full stay-based softmax transition row.

It intentionally stops before cashflow simulation so coefficient parsing,
feature replay, and transition row assembly can be verified independently.

Run from the quantbullet repo root:

    .\\.venv\\Scripts\\python.exe docs/loan_simulation/gam_tieout.py \\
        --roll-rate-root C:/path/to/roll-rate-model

The output workbook is written next to this script by default and is ignored by
git via this directory's .gitignore.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from quantbullet.loan_simulation import Loan, StatusConfig
from quantbullet.loan_simulation.adapters import (
    build_softmax_transition_model,
    parse_rollrate_coefficients,
)


TERMINAL_STATUSES = {"PIF", "LIQ"}
DELINQUENCY_BUCKETS = {
    "D1M": "dq30_balance",
    "D2M": "dq60_balance",
    "D3M": "dq90_balance",
    "D4M": "dq120_balance",
}


def build_feature_rows(n_loans: int, seed: int) -> list[dict[str, Any]]:
    """Create deterministic feature rows covering roll-rate coefficient terms."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(1, n_loans + 1):
        term = str(rng.choice(["36", "60"]))
        month_group = str(rng.choice(["30_Day", "31_Day"]))
        rows.append(
            {
                "loan_id": f"L{index:04d}",
                "employed_f": str(rng.choice(["Yes", "Missing", "No"])),
                "hm_owner": str(rng.choice(["Own", "Mortgage", "Missing", "Rent"])),
                "month": str(
                    rng.choice(
                        [
                            "January",
                            "February",
                            "March",
                            "April",
                            "May",
                            "June",
                            "July",
                            "August",
                            "September",
                            "October",
                            "November",
                            "December",
                        ]
                    )
                ),
                "purpose": str(
                    rng.choice(
                        [
                            "Business",
                            "CC",
                            "Home Improvement",
                            "Medical",
                            "Missing",
                            "Vacation",
                            "Vehicle",
                            "Other",
                        ]
                    )
                ),
                "opti": float(rng.uniform(-0.02, 0.25)),
                "v_opti": float(rng.choice([0.0, 0.5, 1.0])),
                "ofico": float(rng.uniform(540, 850)),
                "v_ofico": float(rng.choice([0.0, 1.0])),
                "rel_fico_ratio_ALL": float(rng.uniform(0.75, 1.25)),
                "v_rel_fico_ratio_ALL": float(rng.choice([0.0, 1.0])),
                "c_credit_age": float(rng.uniform(0, 480)),
                "credit_age": float(rng.uniform(0, 480)),
                "v_credit_age": float(rng.choice([0.0, 0.5, 1.0])),
                "cpi_inflator_12": float(rng.uniform(0.9, 1.2)),
                "cpi_inflator_36": float(rng.uniform(0.9, 1.2)),
                "lending_environment": float(rng.uniform(2014, 2028)),
                "adj_balance_cpi": float(rng.uniform(1_000, 90_000)),
                "c_age_pct": float(rng.uniform(0, 1.25)),
                "oterm_f": term,
                "days_to_month_end": float(rng.uniform(0, 31)),
                "month_group": month_group,
                "rate_incentive_ALL": float(rng.uniform(-0.08, 0.12)),
                "v_rate_incentive_ALL": float(rng.choice([0.0, 1.0])),
            }
        )
    return rows


def build_status_config(status_to_roll: dict[str, list[str]]) -> StatusConfig:
    valid_statuses = set(status_to_roll)
    for row_statuses in status_to_roll.values():
        valid_statuses.update(row_statuses)

    terminal_statuses = valid_statuses & TERMINAL_STATUSES
    return StatusConfig(
        valid_statuses=valid_statuses,
        terminal_statuses=terminal_statuses,
        prepay_statuses={"PIF"} & terminal_statuses,
        default_statuses={"LIQ"} & terminal_statuses,
        delinquency_buckets={
            status: bucket
            for status, bucket in DELINQUENCY_BUCKETS.items()
            if status in valid_statuses
        },
    )


def build_feature_dict(context) -> dict[str, Any]:
    return dict(context.loan.metadata)


def compare_logits(
    feature_rows: list[dict[str, Any]],
    *,
    from_status: str,
    rollrate_models: dict[str, Any],
    quantbullet_models: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for feature_row in feature_rows:
        features = pd.DataFrame([feature_row])
        for to_status, quantbullet_model in sorted(quantbullet_models.items()):
            model_name = f"from{from_status}_{to_status}"
            rollrate_logit = rollrate_models[model_name]
            rollrate_value = rollrate_calc(rollrate_logit, feature_row)
            quantbullet_value = float(quantbullet_model.predict(features)[0])
            rows.append(
                {
                    "loan_id": feature_row["loan_id"],
                    "from_status": from_status,
                    "to_status": to_status,
                    "rollrate_logit": rollrate_value,
                    "quantbullet_logit": quantbullet_value,
                    "logit_diff": quantbullet_value - rollrate_value,
                    "abs_logit_diff": abs(quantbullet_value - rollrate_value),
                }
            )
    return pd.DataFrame(rows)


def compare_probabilities(
    feature_rows: list[dict[str, Any]],
    *,
    from_status: str,
    status_to_roll: dict[str, list[str]],
    rollrate_dm: SimpleNamespace,
    quantbullet_model,
) -> pd.DataFrame:
    roll_to = status_to_roll[from_status]
    rows = []
    for feature_row in feature_rows:
        _, rollrate_probabilities = rollrate_softmax_transition(
            feature_row,
            from_status,
            roll_to,
            rollrate_dm,
            0,
            random.Random(0),
        )
        loan = _feature_row_to_loan(feature_row, from_status)
        quantbullet_probabilities = quantbullet_model.predict(loan, loan.initial_state())

        for to_status, rollrate_probability in zip(roll_to, rollrate_probabilities):
            quantbullet_probability = float(
                quantbullet_probabilities.get(to_status, 0.0)
            )
            rows.append(
                {
                    "loan_id": feature_row["loan_id"],
                    "from_status": from_status,
                    "to_status": to_status,
                    "rollrate_probability": rollrate_probability,
                    "quantbullet_probability": quantbullet_probability,
                    "probability_diff": quantbullet_probability - rollrate_probability,
                    "abs_probability_diff": abs(
                        quantbullet_probability - rollrate_probability
                    ),
                }
            )
    return pd.DataFrame(rows)


def _feature_row_to_loan(feature_row: dict[str, Any], from_status: str) -> Loan:
    return Loan(
        loan_id=str(feature_row["loan_id"]),
        balance=10_000.0,
        annual_rate=0.10,
        term_months=36,
        status=from_status,
        metadata=feature_row,
    )


def build_summary(
    logits: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    logit_summary = (
        logits.groupby("from_status", as_index=False)
        .agg(
            logit_rows=("abs_logit_diff", "size"),
            max_abs_logit_diff=("abs_logit_diff", "max"),
        )
    )
    probability_summary = (
        probabilities.groupby("from_status", as_index=False)
        .agg(
            probability_rows=("abs_probability_diff", "size"),
            max_abs_probability_diff=("abs_probability_diff", "max"),
        )
    )
    summary = logit_summary.merge(
        probability_summary,
        on="from_status",
        how="outer",
    )
    overall = pd.DataFrame(
        [
            {
                "from_status": "ALL",
                "logit_rows": len(logits),
                "max_abs_logit_diff": float(logits["abs_logit_diff"].max()),
                "probability_rows": len(probabilities),
                "max_abs_probability_diff": float(
                    probabilities["abs_probability_diff"].max()
                ),
            },
        ]
    )
    return pd.concat([summary, overall], ignore_index=True)


def write_excel(
    output_path: Path,
    *,
    run_config: pd.DataFrame,
    features: pd.DataFrame,
    logits: pd.DataFrame,
    probabilities: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        run_config.to_excel(writer, sheet_name="run_config", index=False)
        logits.to_excel(writer, sheet_name="logit_diff", index=False)
        probabilities.to_excel(writer, sheet_name="probability_diff", index=False)
        features.to_excel(writer, sheet_name="synthetic_features", index=False)


def load_rollrate_references(
    roll_rate_root: Path,
    *,
    coef_set: str,
    from_status: str,
):
    _add_rollrate_python_path(roll_rate_root)

    global rollrate_calc
    global rollrate_softmax_transition

    from simengine.data_prep import (  # noqa: PLC0415
        build_all_models,
        calc,
        read_coef_file,
    )
    from simengine.runner import (  # noqa: PLC0415
        _build_transition_layout,
        _softmax_transition,
    )

    rollrate_calc = calc
    rollrate_softmax_transition = _softmax_transition

    status_to_roll = _load_status_to_roll(roll_rate_root)
    coef_path = roll_rate_root / "input" / "coef" / coef_set / f"from{from_status}.txt"
    rollrate_models = build_all_models({from_status: read_coef_file(str(coef_path))})
    rollrate_dm = SimpleNamespace(
        models=rollrate_models,
        status_to_roll=status_to_roll,
        clean_status_dict={
            status: status.split(".")[0]
            for statuses in status_to_roll.values()
            for status in statuses
        }
        | {status: status.split(".")[0] for status in status_to_roll},
        prob_layout={},
        dial_data={},
    )
    rollrate_dm._transition_layout = _build_transition_layout(rollrate_dm)
    return coef_path, status_to_roll, rollrate_models, rollrate_dm


def _load_status_to_roll(roll_rate_root: Path) -> dict[str, list[str]]:
    config_path = roll_rate_root / "config" / "default.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    return {
        status: list(roll_to)
        for status, roll_to in config["status_to_roll"].items()
    }


def _add_rollrate_python_path(roll_rate_root: Path) -> None:
    python_dir = roll_rate_root / "python"
    if not python_dir.exists():
        raise FileNotFoundError(f"roll-rate-model python directory not found: {python_dir}")
    sys.path.insert(0, str(python_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare roll-rate GAM logits and softmax rows to quantbullet."
    )
    parser.add_argument("--coef-set", default="GENERIC_v4")
    parser.add_argument(
        "--from-status",
        default="ALL",
        help="From-status to tie out, or ALL for every configured source status.",
    )
    parser.add_argument("--n-loans", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--roll-rate-root",
        type=Path,
        default=None,
        help="Path to roll-rate-model. Can also use ROLL_RATE_MODEL_ROOT.",
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


def _resolve_from_statuses(
    from_status: str,
    status_to_roll: dict[str, list[str]],
) -> list[str]:
    if from_status.upper() == "ALL":
        return list(status_to_roll)
    if from_status not in status_to_roll:
        raise ValueError(
            f"Unknown from_status {from_status!r}; expected ALL or one of "
            f"{sorted(status_to_roll)}"
        )
    return [from_status]


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    roll_rate_root = _resolve_roll_rate_root(args.roll_rate_root)
    status_to_roll = _load_status_to_roll(roll_rate_root)
    from_statuses = _resolve_from_statuses(args.from_status, status_to_roll)
    feature_rows = build_feature_rows(args.n_loans, args.seed)
    features = pd.DataFrame(feature_rows)
    logit_frames = []
    probability_frames = []
    coef_paths = []
    for from_status in from_statuses:
        coef_path, status_to_roll, rollrate_models, rollrate_dm = (
            load_rollrate_references(
                roll_rate_root,
                coef_set=args.coef_set,
                from_status=from_status,
            )
        )
        coef_paths.append(str(coef_path))
        quantbullet_models = parse_rollrate_coefficients(coef_path)
        quantbullet_transition_model = build_softmax_transition_model(
            {from_status: quantbullet_models},
            feature_builder=build_feature_dict,
            status_config=build_status_config(status_to_roll),
        )
        logit_frames.append(
            compare_logits(
                feature_rows,
                from_status=from_status,
                rollrate_models=rollrate_models,
                quantbullet_models=quantbullet_models,
            )
        )
        probability_frames.append(
            compare_probabilities(
                feature_rows,
                from_status=from_status,
                status_to_roll=status_to_roll,
                rollrate_dm=rollrate_dm,
                quantbullet_model=quantbullet_transition_model,
            )
        )

    logits = pd.concat(logit_frames, ignore_index=True)
    probabilities = pd.concat(probability_frames, ignore_index=True)
    summary = build_summary(logits, probabilities)
    elapsed = time.perf_counter() - start
    run_config = pd.DataFrame(
        [
            {"key": "roll_rate_root", "value": str(roll_rate_root)},
            {"key": "coef_paths", "value": "; ".join(coef_paths)},
            {"key": "coef_set", "value": args.coef_set},
            {"key": "from_statuses", "value": ", ".join(from_statuses)},
            {"key": "n_loans", "value": args.n_loans},
            {"key": "seed", "value": args.seed},
            {"key": "elapsed_seconds", "value": round(elapsed, 3)},
        ]
    )
    write_excel(
        args.output,
        run_config=run_config,
        features=features,
        logits=logits,
        probabilities=probabilities,
        summary=summary,
    )

    print(summary.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
