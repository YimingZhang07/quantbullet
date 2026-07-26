"""Period-1 transition benchmark for PAR_2025_1 + GENERIC_v4."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from feature_builder import build_feature_dict, enrich_feature_rows
from quantbullet.loan_simulation import Loan, StatusConfig
from quantbullet.loan_simulation.adapters import (
    build_softmax_transition_model,
    parse_rollrate_coefficients,
)


EXAMPLE_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_prepped_loans(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        loans = json.load(file)
    if not isinstance(loans, list):
        raise ValueError(f"Expected a list of loans in {path}")
    return loans


def build_status_config(config: dict[str, Any]) -> StatusConfig:
    return StatusConfig(
        valid_statuses=_status_universe(config["status_to_roll"]),
        terminal_statuses=set(config["terminal_statuses"]),
        prepay_statuses=set(config.get("prepay_statuses", [])),
        default_statuses=set(config.get("default_statuses", [])),
        delinquency_buckets=dict(config.get("delinquency_buckets", {})),
    )


def _status_universe(status_to_roll: dict[str, list[str]]) -> set[str]:
    statuses = set(status_to_roll)
    for row_statuses in status_to_roll.values():
        statuses.update(row_statuses)
    return statuses


def load_edge_models(
    coefficient_root: Path,
    *,
    from_statuses: list[str],
) -> dict[str, dict[str, Any]]:
    return {
        from_status: parse_rollrate_coefficients(
            coefficient_root / f"from{from_status}.txt"
        )
        for from_status in from_statuses
    }


def load_rollrate_references(
    roll_rate_root: Path,
    *,
    coefficient_root: Path,
    from_statuses: list[str],
    status_to_roll: dict[str, list[str]],
):
    _add_rollrate_python_path(roll_rate_root)

    global rollrate_calc
    global rollrate_softmax_transition

    from simengine.data_prep import build_all_models, calc, read_coef_file  # noqa: PLC0415
    from simengine.runner import _build_transition_layout, _softmax_transition  # noqa: PLC0415

    rollrate_calc = calc
    rollrate_softmax_transition = _softmax_transition

    coef_by_from = {
        from_status: read_coef_file(str(coefficient_root / f"from{from_status}.txt"))
        for from_status in from_statuses
    }
    rollrate_models = build_all_models(coef_by_from)
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
    return rollrate_models, rollrate_dm


def _add_rollrate_python_path(roll_rate_root: Path) -> None:
    python_dir = roll_rate_root / "python"
    if not python_dir.exists():
        raise FileNotFoundError(f"roll-rate-model python directory not found: {python_dir}")
    sys.path.insert(0, str(python_dir))


def compare_logits(
    feature_rows: list[dict[str, Any]],
    *,
    from_statuses: list[str],
    rollrate_models: dict[str, Any],
    quantbullet_models: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    active_from_statuses = set(from_statuses)
    for feature_row in feature_rows:
        from_status = str(feature_row["status"])
        if from_status not in active_from_statuses:
            continue
        features = pd.DataFrame([feature_row])
        for to_status, quantbullet_model in sorted(quantbullet_models[from_status].items()):
            model_name = f"from{from_status}_{to_status}"
            rollrate_value = rollrate_calc(rollrate_models[model_name], feature_row)
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
    status_to_roll: dict[str, list[str]],
    rollrate_dm: SimpleNamespace,
    quantbullet_transition_model,
) -> pd.DataFrame:
    rows = []
    for feature_row in feature_rows:
        from_status = str(feature_row["status"])
        if from_status not in status_to_roll:
            continue
        roll_to = status_to_roll[from_status]
        _, rollrate_probabilities = rollrate_softmax_transition(
            feature_row,
            from_status,
            roll_to,
            rollrate_dm,
            0,
            random.Random(0),
        )
        loan = _feature_row_to_loan(feature_row)
        quantbullet_probabilities = quantbullet_transition_model.predict(
            loan,
            loan.initial_state(),
        )
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


def _feature_row_to_loan(feature_row: dict[str, Any]) -> Loan:
    scheduled_payment = feature_row.get("loan_payment_scheduled_amt_orig")
    return Loan(
        loan_id=str(feature_row["loan_id"]),
        balance=max(float(feature_row["end_bal"]), 0.0),
        annual_rate=float(feature_row.get("int_rate", feature_row["note_rate"])),
        term_months=int(feature_row["term"]),
        original_balance=float(feature_row["orig_bal"]),
        age_months=int(feature_row.get("loan_age", feature_row.get("age", 0))),
        status=str(feature_row["status"]),
        scheduled_payment=(
            float(scheduled_payment) if scheduled_payment is not None else None
        ),
        metadata=feature_row,
    )


def build_summary(logits: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
    logit_summary = (
        logits.groupby("from_status", as_index=False)
        .agg(
            loan_rows=("loan_id", "nunique"),
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
    summary = logit_summary.merge(probability_summary, on="from_status", how="outer")
    overall = pd.DataFrame(
        [
            {
                "from_status": "ALL",
                "loan_rows": probabilities["loan_id"].nunique(),
                "logit_rows": len(logits),
                "max_abs_logit_diff": float(logits["abs_logit_diff"].max()),
                "probability_rows": len(probabilities),
                "max_abs_probability_diff": float(
                    probabilities["abs_probability_diff"].max()
                ),
            }
        ]
    )
    return pd.concat([summary, overall], ignore_index=True)


def write_excel(
    output_path: Path,
    *,
    summary: pd.DataFrame,
    logits: pd.DataFrame,
    probabilities: pd.DataFrame,
    enriched_loans: pd.DataFrame,
    run_config: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        run_config.to_excel(writer, sheet_name="run_config", index=False)
        probabilities.to_excel(writer, sheet_name="probability_diff", index=False)
        logits.to_excel(writer, sheet_name="logit_diff", index=False)
        enriched_loans.to_excel(writer, sheet_name="enriched_loans", index=False)


def run_benchmark(
    config: dict[str, Any],
    *,
    roll_rate_root: Path,
    output_path: Path,
) -> pd.DataFrame:
    coefficient_root = (
        roll_rate_root / "input" / "coef" / config["coefficient_set"]
    )
    raw_loans = load_prepped_loans(roll_rate_root / config["loans_prepped_path"])
    enriched_loans = enrich_feature_rows(raw_loans)
    edge_models = load_edge_models(
        coefficient_root,
        from_statuses=list(config["from_statuses"]),
    )
    rollrate_models, rollrate_dm = load_rollrate_references(
        roll_rate_root,
        coefficient_root=coefficient_root,
        from_statuses=list(config["from_statuses"]),
        status_to_roll=config["status_to_roll"],
    )
    quantbullet_transition_model = build_softmax_transition_model(
        edge_models,
        feature_builder=build_feature_dict,
        status_config=build_status_config(config),
    )
    logits = compare_logits(
        enriched_loans,
        from_statuses=list(config["from_statuses"]),
        rollrate_models=rollrate_models,
        quantbullet_models=edge_models,
    )
    probabilities = compare_probabilities(
        enriched_loans,
        status_to_roll=config["status_to_roll"],
        rollrate_dm=rollrate_dm,
        quantbullet_transition_model=quantbullet_transition_model,
    )
    summary = build_summary(logits, probabilities)
    run_config = pd.DataFrame(
        [
            {"key": "deal_name", "value": config["deal_name"]},
            {"key": "coefficient_set", "value": config["coefficient_set"]},
            {"key": "loan_count", "value": len(enriched_loans)},
            {"key": "roll_rate_root", "value": str(roll_rate_root)},
            {"key": "coefficient_root", "value": str(coefficient_root)},
        ]
    )
    write_excel(
        output_path,
        summary=summary,
        logits=logits,
        probabilities=probabilities,
        enriched_loans=pd.DataFrame(enriched_loans),
        run_config=run_config,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run period-1 transition benchmark for PAR_2025_1 + GENERIC_v4."
    )
    parser.add_argument("--config", type=Path, default=EXAMPLE_DIR / "config.json")
    parser.add_argument(
        "--roll-rate-root",
        type=Path,
        default=None,
        help="Path to roll-rate-model. Can also use ROLL_RATE_MODEL_ROOT.",
    )
    parser.add_argument("--output", type=Path, default=None)
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
    config = load_config(args.config)
    output_path = args.output or (EXAMPLE_DIR / "period1_transition.xlsx")
    summary = run_benchmark(
        config,
        roll_rate_root=_resolve_roll_rate_root(args.roll_rate_root),
        output_path=output_path,
    )
    print(summary.to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
