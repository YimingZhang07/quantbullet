"""Inventory coefficient-required features against a prepared deal file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from feature_builder import enrich_feature_rows
from quantbullet.loan_simulation.adapters import parse_rollrate_coefficients
from quantbullet.model.gam.terms import (
    FactorTermData,
    SplineByGroupTermData,
    SplineByNumericTermData,
    SplineTermData,
    TensorTermData,
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


def coefficient_feature_usage(
    coefficient_root: Path,
    *,
    from_statuses: list[str],
) -> pd.DataFrame:
    rows = []
    for from_status in from_statuses:
        edge_models = parse_rollrate_coefficients(
            coefficient_root / f"from{from_status}.txt"
        )
        for to_status, edge_model in sorted(edge_models.items()):
            for feature in sorted(required_features(edge_model)):
                rows.append(
                    {
                        "feature": feature,
                        "from_status": from_status,
                        "to_status": to_status,
                        "edge": f"{from_status}->{to_status}",
                    }
                )
    return pd.DataFrame(rows)


def required_features(edge_model) -> set[str]:
    features: set[str] = set()
    for term in edge_model.term_data.values():
        if isinstance(term, (FactorTermData, SplineTermData)):
            features.add(term.feature)
        elif isinstance(term, SplineByGroupTermData):
            features.add(term.feature)
            features.add(term.by_feature)
        elif isinstance(term, SplineByNumericTermData):
            features.add(term.feature)
            features.add(term.multiplier_feature)
        elif isinstance(term, TensorTermData):
            features.add(term.feature_x)
            features.add(term.feature_y)
        else:
            raise ValueError(f"Unsupported term type: {type(term)}")
    return features


def prepared_field_inventory(loans: list[dict[str, Any]]) -> pd.DataFrame:
    all_fields = sorted({field for loan in loans for field in loan})
    rows = []
    for field in all_fields:
        values = [loan.get(field) for loan in loans]
        non_null_values = [value for value in values if value is not None]
        rows.append(
            {
                "field": field,
                "non_null_count": len(non_null_values),
                "null_count": len(values) - len(non_null_values),
                "sample_values": _sample_values(non_null_values),
            }
        )
    return pd.DataFrame(rows)


def build_required_feature_inventory(
    usage: pd.DataFrame,
    loans: list[dict[str, Any]],
) -> pd.DataFrame:
    prepared_fields = {field for loan in loans for field in loan}
    usage_by_feature = (
        usage.groupby("feature")
        .agg(
            from_statuses=("from_status", lambda values: ", ".join(sorted(set(values)))),
            to_statuses=("to_status", lambda values: ", ".join(sorted(set(values)))),
            edges=("edge", lambda values: ", ".join(sorted(set(values)))),
        )
        .reset_index()
    )
    rows = []
    for _, usage_row in usage_by_feature.iterrows():
        feature = usage_row["feature"]
        values = [loan.get(feature) for loan in loans]
        non_null_values = [value for value in values if value is not None]
        rows.append(
            {
                "feature": feature,
                "feature_group": classify_feature(feature),
                "present_in_prepped": feature in prepared_fields,
                "non_null_count": len(non_null_values) if feature in prepared_fields else 0,
                "null_count": (
                    len(values) - len(non_null_values)
                    if feature in prepared_fields
                    else len(loans)
                ),
                "sample_values": _sample_values(non_null_values),
                "from_statuses": usage_row["from_statuses"],
                "to_statuses": usage_row["to_statuses"],
                "edges": usage_row["edges"],
            }
        )
    return pd.DataFrame(rows).sort_values(["present_in_prepped", "feature"])


def classify_feature(feature: str) -> str:
    if feature.startswith("v_"):
        return "validity_flag"
    if feature in {
        "cpi_inflator_12",
        "cpi_inflator_36",
        "lending_environment",
        "rate_incentive_ALL",
    }:
        return "macro_or_market"
    if feature in {
        "age",
        "age_pct",
        "c_age_pct",
        "credit_age",
        "c_credit_age",
        "days_to_month_end",
        "loan_age",
        "month",
        "month_group",
    }:
        return "time_varying_or_period"
    return "static_or_prepped_metadata"


def build_summary(
    required_inventory: pd.DataFrame,
    available_inventory: pd.DataFrame,
    loans: list[dict[str, Any]],
) -> pd.DataFrame:
    missing = required_inventory[~required_inventory["present_in_prepped"]]
    partially_null = required_inventory[
        required_inventory["present_in_prepped"] & (required_inventory["null_count"] > 0)
    ]
    return pd.DataFrame(
        [
            {"metric": "loan_count", "value": len(loans)},
            {"metric": "available_field_count", "value": len(available_inventory)},
            {"metric": "required_feature_count", "value": len(required_inventory)},
            {"metric": "missing_required_feature_count", "value": len(missing)},
            {"metric": "partially_null_required_feature_count", "value": len(partially_null)},
        ]
    )


def _sample_values(values: list[Any], limit: int = 5) -> str:
    seen = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.append(text)
        if len(seen) >= limit:
            break
    return ", ".join(seen)


def write_excel(
    output_path: Path,
    *,
    summary: pd.DataFrame,
    required_features_frame: pd.DataFrame,
    feature_usage: pd.DataFrame,
    available_fields: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        required_features_frame.to_excel(
            writer,
            sheet_name="required_features",
            index=False,
        )
        feature_usage.to_excel(writer, sheet_name="feature_usage", index=False)
        available_fields.to_excel(writer, sheet_name="available_fields", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory GENERIC_v4 coefficient features against PAR_2025_1 prepared loans."
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
    roll_rate_root = _resolve_roll_rate_root(args.roll_rate_root)
    coefficient_root = (
        roll_rate_root / "input" / "coef" / config["coefficient_set"]
    )
    raw_loans = load_prepped_loans(roll_rate_root / config["loans_prepped_path"])
    loans = enrich_feature_rows(raw_loans, config)
    usage = coefficient_feature_usage(
        coefficient_root,
        from_statuses=list(config["from_statuses"]),
    )
    available_fields = prepared_field_inventory(loans)
    required_features_frame = build_required_feature_inventory(usage, loans)
    summary = build_summary(required_features_frame, available_fields, loans)
    output_path = args.output or (EXAMPLE_DIR / config["output"])
    write_excel(
        output_path,
        summary=summary,
        required_features_frame=required_features_frame,
        feature_usage=usage,
        available_fields=available_fields,
    )

    print(summary.to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
