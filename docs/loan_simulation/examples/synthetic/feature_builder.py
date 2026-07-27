"""Build the three model features used by the synthetic example."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from quantbullet.loan_simulation import (
    FeatureContext,
    Loan,
    LoanState,
    PeriodCashflow,
    RuntimeFeatureProvider,
)


MODEL_FEATURE_NAMES = ("age", "incentive", "hpi")


def build_feature_dict(context: FeatureContext) -> Mapping[str, float]:
    """Return only the declared model features from a transition context."""
    return {
        feature_name: float(context.model_features[feature_name])
        for feature_name in MODEL_FEATURE_NAMES
    }


class SyntheticFeatureProvider(RuntimeFeatureProvider):
    """Derive model features from loan state and monthly macro inputs."""

    def initialize_path_state(
        self,
        loan: Loan,
        start_period: pd.Period,
    ) -> None:
        """Initialize no path-local state because all features are current-period."""
        del loan, start_period
        return None

    def model_features_for_period(
        self,
        *,
        loan: Loan,
        current_state: LoanState,
        period_date: pd.Period,
        macro_features: Mapping[str, Any],
        path_features: Mapping[str, Any],
        feature_state: Any,
    ) -> Mapping[str, float]:
        del period_date, path_features, feature_state
        market_rate = float(macro_features["market_rate"])
        return {
            "age": float(current_state.age_months),
            "incentive": float(loan.annual_rate) - market_rate,
            "hpi": float(macro_features["hpi"]),
        }

    def advance_path_state(
        self,
        *,
        feature_state: Any,
        cashflow: PeriodCashflow,
        next_state: LoanState,
    ) -> None:
        """Advance no state because every feature is rebuilt for each period."""
        del feature_state, cashflow, next_state
        return None
