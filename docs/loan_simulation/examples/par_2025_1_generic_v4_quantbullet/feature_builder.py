"""Feature builder for the PAR_2025_1 + GENERIC_v4 QuantBullet demo."""

from __future__ import annotations

import calendar
from typing import Any

from quantbullet.loan_simulation import RuntimeFeatureProvider


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


def build_feature_dict(context) -> dict[str, Any]:
    return dict(context.model_features)


class GenericV4FeatureProvider(RuntimeFeatureProvider):
    """Per-path runtime features for this GENERIC_v4 prepared-loan demo."""

    def initialize_path_state(self, loan, start_period):
        feature_state = dict(loan.metadata)
        feature_state["days_to_month_end"] = derive_days_to_month_end(feature_state)
        feature_state["month_group"] = derive_month_group(
            feature_state["days_to_month_end"]
        )
        start_year, start_month = _parse_year_month(feature_state["r_dt"])
        feature_state["_start_year"] = start_year
        feature_state["_start_month"] = start_month
        return feature_state

    def model_features_for_period(
        self,
        *,
        loan,
        current_state,
        period_date,
        macro_features,
        path_features,
        feature_state,
    ):
        features = dict(feature_state)
        features["status"] = current_state.status
        features["end_bal"] = current_state.balance
        return features

    def advance_path_state(self, *, feature_state, cashflow, next_state):
        feature_state["loan_age"] = int(feature_state.get("loan_age", 0)) + 1
        feature_state["age"] = feature_state["loan_age"]
        term = float(feature_state.get("term", 1))
        feature_state["age_pct"] = (
            float(feature_state["loan_age"]) / term if term != 0 else 0.0
        )
        feature_state["c_age_pct"] = feature_state["age_pct"]

        year, month = _advance_month(
            int(feature_state["_start_year"]),
            int(feature_state["_start_month"]),
            next_state.period,
        )
        feature_state["r_dt"] = _end_of_month(year, month)
        feature_state["month"] = MONTHS[month - 1]
        feature_state["days_to_month_end"] = derive_days_to_month_end(feature_state)
        feature_state["month_group"] = derive_month_group(
            feature_state["days_to_month_end"]
        )


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
