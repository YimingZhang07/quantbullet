from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from .entities import Loan, LoanState


class SeverityProvider(ABC):
    """Base class for loss severity assumptions or models."""

    @abstractmethod
    def severity(
        self,
        loan: Loan,
        begin_state: LoanState,
        end_status: str,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
    ) -> float:
        """Return loss severity as a decimal between 0 and 1."""
        raise NotImplementedError


class ConstantSeverityProvider(SeverityProvider):
    """Constant loss severity provider."""

    def __init__(self, severity: float) -> None:
        severity = float(severity)
        if severity < 0 or severity > 1:
            raise ValueError("severity must be between 0 and 1")
        self._severity = severity

    def severity(
        self,
        loan: Loan,
        begin_state: LoanState,
        end_status: str,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
    ) -> float:
        return self._severity


class RecoveryLagProvider(ABC):
    """Base class for recovery timing assumptions or models."""

    @abstractmethod
    def recovery_lag_periods(
        self,
        loan: Loan,
        begin_state: LoanState,
        end_status: str,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
    ) -> int:
        """Return the number of periods between default and recovery receipt."""
        raise NotImplementedError


class ConstantRecoveryLagProvider(RecoveryLagProvider):
    """Constant recovery lag provider."""

    def __init__(self, lag_periods: int) -> None:
        lag_periods = int(lag_periods)
        if lag_periods < 0:
            raise ValueError("lag_periods must be non-negative")
        self._lag_periods = lag_periods

    def recovery_lag_periods(
        self,
        loan: Loan,
        begin_state: LoanState,
        end_status: str,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
    ) -> int:
        return self._lag_periods
