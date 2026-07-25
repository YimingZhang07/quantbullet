"""Tie out run-level runtime feature updates against roll-rate registry updates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from feature_builder import init_runtime_feature_state, step_runtime_features


EXAMPLE_DIR = Path(__file__).resolve().parent
COMPARE_FIELDS = [
    "r_dt",
    "loan_age",
    "age",
    "age_pct",
    "c_age_pct",
    "month",
    "days_to_month_end",
    "month_group",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_prepped_loans(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        loans = json.load(file)
    if not isinstance(loans, list):
        raise ValueError(f"Expected a list of loans in {path}")
    return loans


def compare_runtime_features(
    loans: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    horizon: int,
    roll_rate_root: Path,
) -> pd.DataFrame:
    _add_rollrate_python_path(roll_rate_root)
    from simengine.data_prep import init_time_varying_state, step_period_fields  # noqa: PLC0415

    rows = []
    for loan in loans:
        our_state = init_runtime_feature_state(loan, config)
        rollrate_state = dict(loan)
        init_time_varying_state(rollrate_state)
        rollrate_state = init_runtime_feature_state(rollrate_state, config)

        rows.extend(_comparison_rows(loan["loan_id"], 0, our_state, rollrate_state))
        for period in range(1, horizon + 1):
            our_state = step_runtime_features(
                our_state,
                next_period=period,
                config=config,
            )
            step_period_fields(rollrate_state, period)
            rows.extend(
                _comparison_rows(
                    loan["loan_id"],
                    period,
                    our_state,
                    rollrate_state,
                )
            )
    return pd.DataFrame(rows)


def _comparison_rows(
    loan_id: Any,
    period: int,
    our_state: dict[str, Any],
    rollrate_state: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for field in COMPARE_FIELDS:
        our_value = our_state.get(field)
        rollrate_value = rollrate_state.get(field)
        rows.append(
            {
                "loan_id": loan_id,
                "period": period,
                "field": field,
                "our_value": our_value,
                "rollrate_value": rollrate_value,
                "matches": _values_match(our_value, rollrate_value),
            }
        )
    return rows


def _values_match(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) < 1e-12
        except (TypeError, ValueError):
            return False
    return left == right


def build_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    by_field = (
        comparison.groupby("field", as_index=False)
        .agg(
            rows=("matches", "size"),
            mismatches=("matches", lambda values: int((~values).sum())),
        )
    )
    overall = pd.DataFrame(
        [
            {
                "field": "ALL",
                "rows": len(comparison),
                "mismatches": int((~comparison["matches"]).sum()),
            }
        ]
    )
    return pd.concat([by_field, overall], ignore_index=True)


def write_excel(
    output_path: Path,
    *,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    mismatches = comparison[~comparison["matches"]]
    with pd.ExcelWriter(output_path) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        mismatches.to_excel(writer, sheet_name="mismatches", index=False)
        comparison.to_excel(writer, sheet_name="comparison", index=False)


def _add_rollrate_python_path(roll_rate_root: Path) -> None:
    python_dir = roll_rate_root / "python"
    if not python_dir.exists():
        raise FileNotFoundError(f"roll-rate-model python directory not found: {python_dir}")
    sys.path.insert(0, str(python_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tie out run-level runtime feature updates against roll-rate."
    )
    parser.add_argument("--config", type=Path, default=EXAMPLE_DIR / "config.json")
    parser.add_argument(
        "--roll-rate-root",
        type=Path,
        default=None,
        help="Path to roll-rate-model. Can also use ROLL_RATE_MODEL_ROOT.",
    )
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-loans", type=int, default=100)
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
    roll_rate_root = _resolve_roll_rate_root(args.roll_rate_root)
    loans = load_prepped_loans(roll_rate_root / config["loans_prepped_path"])
    loans = loans[: args.max_loans]
    horizon = args.horizon or int(config["runtime_feature_horizon"])
    comparison = compare_runtime_features(
        loans,
        config=config,
        horizon=horizon,
        roll_rate_root=roll_rate_root,
    )
    summary = build_summary(comparison)
    output_path = args.output or (EXAMPLE_DIR / "runtime_feature_tieout.xlsx")
    write_excel(output_path, summary=summary, comparison=comparison)
    print(summary.to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
