from __future__ import annotations

from abc import ABC
from typing import Any, Mapping

import pandas as pd

from .entities import Loan, LoanState, PeriodCashflow


class RuntimeFeatureProvider(ABC):
    """Base hook for per-path model feature state managed by ``LoanSimulator``."""

    def initialize_path_state(self, loan: Loan, start_period: pd.Period) -> Any:
        """Create per-path feature state at path start."""
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
    ) -> Mapping[str, Any]:
        """Return model features for the current transition period."""
        return {}

    def advance_path_state(
        self,
        *,
        feature_state: Any,
        cashflow: PeriodCashflow,
        next_state: LoanState,
    ) -> None:
        """Update feature state after cashflow projection."""
        return None


class EmptyRuntimeFeatureProvider(RuntimeFeatureProvider):
    """Default provider for simulations that do not need runtime model features."""
