"""Build the three model features used by the synthetic example."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantbullet.loan_simulation import (
    FeatureStateBase,
    Loan,
    LoanState,
    RuntimeFeatureProvider,
)


def advance_age(state: FeatureState, env: Mapping[str, Any]) -> float:
    del env
    return float(state.age_months)


def advance_incentive(state: FeatureState, env: Mapping[str, Any]) -> float:
    return float(state.annual_rate) - float(env["market_rate"])


def advance_hpi(state: FeatureState, env: Mapping[str, Any]) -> float:
    return float(env["hpi"])


@dataclass
class FeatureState(FeatureStateBase):
    """Synthetic model feature state and metadata.

    The fields with ``*_model`` metadata define the model input schema. Context
    fields are provider-owned inputs used to advance those model-facing fields.
    """

    # model-facing fields
    age: float = field(
        init=False,
        metadata={
            "kind": "dynamic_model",
            "deps": ("age_months",),
            "advance": advance_age,
        },
    )
    incentive: float = field(
        init=False,
        metadata={
            "kind": "dynamic_model",
            "deps": ("annual_rate", "market_rate"),
            "advance": advance_incentive,
        },
    )
    hpi: float = field(
        init=False,
        metadata={
            "kind": "dynamic_model",
            "deps": ("hpi",),
            "advance": advance_hpi,
        },
    )

    # provider-only context
    annual_rate: float = field(metadata={"kind": "static_context"})
    age_months: int = field(metadata={"kind": "dynamic_context"})

    def update_for_period(
        self,
        *,
        loan: Loan,
        current_state: LoanState,
        macro_features: Mapping[str, Any],
    ) -> None:
        self.annual_rate = float(loan.annual_rate)
        self.age_months = int(current_state.age_months)
        for spec in self.FEATURE_SPECS:
            if spec.advance is not None:
                setattr(self, spec.name, spec.advance(self, macro_features))


FeatureState.configure_feature_metadata()


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
        state.update_for_period(
            loan=loan,
            current_state=current_state,
            macro_features=macro_features,
        )

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
        state = _require_feature_state(feature_state)
        return state.model_features()


def _require_feature_state(feature_state: Any) -> FeatureState:
    if not isinstance(feature_state, FeatureState):
        raise TypeError(
            "SyntheticFeatureProvider expected FeatureState; "
            f"got {type(feature_state).__name__}"
        )
    return feature_state
