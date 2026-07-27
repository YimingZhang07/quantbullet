"""Build the three model features used by the synthetic example."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from quantbullet.loan_simulation import (
    FeatureStateBase,
    Loan,
    LoanState,
    RuntimeFeatureProvider,
    model_feature,
)


def update_age(state: FeatureState, env: Mapping[str, Any]) -> float:
    del env
    return float(state.age_months)


def update_incentive(state: FeatureState, env: Mapping[str, Any]) -> float:
    return float(state.annual_rate) - float(env["market_rate"])


def update_hpi(state: FeatureState, env: Mapping[str, Any]) -> float:
    return float(env["hpi"])


@dataclass
class FeatureState(FeatureStateBase):
    """Synthetic runtime feature state.

    Plain fields are provider context. ``model_feature`` fields form the model
    input schema, and each one owns its per-period update logic.
    """

    # provider context
    annual_rate: float
    age_months: int

    # model features
    age: float = model_feature(update_age, init=False)
    incentive: float = model_feature(update_incentive, init=False)
    hpi: float = model_feature(update_hpi, init=False)


class SyntheticFeatureProvider(RuntimeFeatureProvider):
    """Derive model features from loan state and monthly macro inputs."""

    def initialize_path_state(
        self,
        loan: Loan,
        start_period: pd.Period,
    ) -> FeatureState:
        """Initialize a per-path feature state with static loan context."""
        del start_period
        return FeatureState(
            annual_rate=float(loan.annual_rate),
            age_months=loan.age_months,
        )

    def prepare_period_state(
        self,
        *,
        loan: Loan,
        current_state: LoanState,
        period_date: pd.Period,
        macro_features: Mapping[str, Any],
        path_features: Mapping[str, Any],
        feature_state: Any,
    ) -> None:
        del period_date, path_features
        state = _require_feature_state(feature_state)
        state.annual_rate = float(loan.annual_rate)
        state.age_months = int(current_state.age_months)
        state.update_features(macro_features)

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
        del loan, current_state, period_date, macro_features, path_features
        return _require_feature_state(feature_state).model_features()


def _require_feature_state(feature_state: Any) -> FeatureState:
    if not isinstance(feature_state, FeatureState):
        raise TypeError(
            "SyntheticFeatureProvider expected FeatureState; "
            f"got {type(feature_state).__name__}"
        )
    return feature_state
