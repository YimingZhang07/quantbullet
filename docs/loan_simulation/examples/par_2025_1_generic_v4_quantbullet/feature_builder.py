"""Feature builder for the PAR_2025_1 + GENERIC_v4 QuantBullet demo."""

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


def enrich_feature_row(feature_row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(feature_row)
    enriched["days_to_month_end"] = derive_days_to_month_end(enriched)
    enriched["month_group"] = derive_month_group(enriched["days_to_month_end"])
    return enriched


def init_runtime_feature_state(feature_row: dict[str, Any]) -> dict[str, Any]:
    state = enrich_feature_row(feature_row)
    start_year, start_month = _parse_year_month(state["r_dt"])
    state["_start_year"] = start_year
    state["_start_month"] = start_month
    return state


def step_runtime_features(
    feature_state: dict[str, Any],
    *,
    next_period: int,
) -> dict[str, Any]:
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
    updated["days_to_month_end"] = derive_days_to_month_end(updated)
    updated["month_group"] = derive_month_group(updated["days_to_month_end"])
    return updated


def build_feature_dict(context) -> dict[str, Any]:
    state = init_runtime_feature_state(dict(context.loan.metadata))
    for period in range(1, context.current_state.period + 1):
        state = step_runtime_features(state, next_period=period)
    state["status"] = context.current_state.status
    state["end_bal"] = context.current_state.balance
    return state


def derive_days_to_month_end(features: dict[str, Any]) -> int:
    year, month = _parse_year_month(features["r_dt"])
    payment_day = int(features.get("pmt_day", 15))
    days_in_month = calendar.monthrange(year, month)[1]
    return days_in_month - min(payment_day, days_in_month)


def derive_month_group(days_to_month_end: int) -> str:
    if days_to_month_end <= 28:
        return "30_Day"
    return "31_Day"


def _parse_year_month(date_value: Any) -> tuple[int, int]:
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
