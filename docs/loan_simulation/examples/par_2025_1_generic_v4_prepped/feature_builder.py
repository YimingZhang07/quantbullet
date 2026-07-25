"""Run-level feature enrichment for PAR_2025_1 + GENERIC_v4."""

from __future__ import annotations

import calendar
from typing import Any


MONTHS = [
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


def enrich_feature_rows(
    feature_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return enriched feature rows without mutating the prepared loan inputs."""
    return [enrich_feature_row(row, config) for row in feature_rows]


def enrich_feature_row(
    feature_row: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(feature_row)
    for feature_name, rule in config.get("derived_features", {}).items():
        rule_type = rule["type"]
        if rule_type == "days_to_month_end":
            enriched[feature_name] = derive_days_to_month_end(enriched, rule)
        elif rule_type == "month_group":
            enriched[feature_name] = derive_month_group(enriched, rule)
        else:
            raise ValueError(f"Unsupported derived feature rule type: {rule_type!r}")
    return enriched


def build_feature_dict(context) -> dict[str, Any]:
    """Map FeatureContext to the enriched feature row stored on Loan.metadata."""
    return dict(context.loan.metadata)


def init_runtime_feature_state(
    feature_row: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Initialize per-path runtime feature state for the prepared feature row."""
    state = enrich_feature_row(feature_row, config)
    start_year, start_month = _parse_year_month(state["r_dt"])
    state["_start_year"] = start_year
    state["_start_month"] = start_month
    return state


def step_runtime_features(
    feature_state: dict[str, Any],
    *,
    next_period: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Advance the run-level dynamic features after a period is evaluated."""
    updated = dict(feature_state)

    updated["loan_age"] = int(updated.get("loan_age", 0)) + 1
    updated["age"] = updated["loan_age"]
    term = float(updated.get("term", 1))
    updated["age_pct"] = float(updated["loan_age"]) / term if term != 0 else 0.0
    updated["c_age_pct"] = updated["age_pct"]

    year, month = _advance_month(
        int(updated["_start_year"]),
        int(updated["_start_month"]),
        next_period,
    )
    updated["r_dt"] = _end_of_month(year, month)
    updated["month"] = MONTHS[month - 1]

    for feature_name, rule in config.get("derived_features", {}).items():
        rule_type = rule["type"]
        if rule_type == "days_to_month_end":
            updated[feature_name] = derive_days_to_month_end(updated, rule)
        elif rule_type == "month_group":
            updated[feature_name] = derive_month_group(updated, rule)
        else:
            raise ValueError(f"Unsupported derived feature rule type: {rule_type!r}")

    return updated


def derive_days_to_month_end(
    features: dict[str, Any],
    rule: dict[str, Any],
) -> int:
    date_value = features.get(rule["date_field"])
    year, month = _parse_year_month(date_value)
    payment_day = int(features.get(rule["payment_day_field"], 15))
    days_in_month = calendar.monthrange(year, month)[1]
    return days_in_month - min(payment_day, days_in_month)


def derive_month_group(
    features: dict[str, Any],
    rule: dict[str, Any],
) -> str:
    value = features[rule["source"]]
    if value <= rule["threshold"]:
        return rule["lte_label"]
    return rule["gt_label"]


def _parse_year_month(date_value: Any) -> tuple[int, int]:
    if date_value is None:
        raise ValueError("date_value is required")
    text = str(date_value).strip()
    if "/" in text:
        parts = text.split("/")
        return int(parts[2]), int(parts[0])
    return int(text[:4]), int(text[5:7])


def _advance_month(year: int, month: int, periods: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + periods
    return total // 12, total % 12 + 1


def _end_of_month(year: int, month: int) -> str:
    day = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-{day:02d}"
