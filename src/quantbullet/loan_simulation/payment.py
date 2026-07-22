from __future__ import annotations

from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Mapping

from .entities import Loan, LoanState
from .status import DEFAULT_STATUS_CONFIG, StatusConfig


class PaymentPolicy(ABC):
    """Base class for scheduled payment collection rules.

    The first-phase policy controls how many scheduled payments are collected
    for a begin-status to end-status transition. Product-specific cashflow
    structures can later use a richer instruction object if needed.
    """

    @abstractmethod
    def payment_periods(
        self,
        loan: Loan,
        begin_state: LoanState,
        end_status: str,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
    ) -> int:
        """Return the number of scheduled payments collected this period."""
        raise NotImplementedError


class MatrixPaymentPolicy(PaymentPolicy):
    """Scheduled payment policy backed by a status transition matrix."""

    def __init__(
        self,
        payment_periods: Mapping[str, Mapping[str, int]],
        *,
        status_config: StatusConfig | None = None,
    ) -> None:
        self.status_config = status_config or DEFAULT_STATUS_CONFIG
        self._payment_periods = _freeze_payment_matrix(
            payment_periods,
            status_config=self.status_config,
        )

    @property
    def payment_matrix(self) -> Mapping[str, Mapping[str, int]]:
        return self._payment_periods

    def payment_periods(
        self,
        loan: Loan,
        begin_state: LoanState,
        end_status: str,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
    ) -> int:
        begin_status = self.status_config.require_valid_status(begin_state.status)
        end_status = self.status_config.require_valid_status(end_status)
        return self._payment_periods[begin_status][end_status]


def _freeze_payment_matrix(
    payment_periods: Mapping[str, Mapping[str, int]],
    *,
    status_config: StatusConfig,
) -> Mapping[str, Mapping[str, int]]:
    if not payment_periods:
        raise ValueError("payment_periods must be non-empty")

    normalized: dict[str, Mapping[str, int]] = {}
    for begin_status, row in payment_periods.items():
        normalized_begin_status = status_config.require_valid_status(begin_status)
        normalized[normalized_begin_status] = _freeze_payment_row(
            row,
            status_config=status_config,
        )

    missing_rows = status_config.valid_statuses - frozenset(normalized)
    if missing_rows:
        raise ValueError(
            "Payment matrix is missing rows for statuses: "
            f"{sorted(missing_rows)}"
        )

    return MappingProxyType(normalized)


def _freeze_payment_row(
    row: Mapping[str, int],
    *,
    status_config: StatusConfig,
) -> Mapping[str, int]:
    if not row:
        raise ValueError("payment matrix rows must be non-empty")

    normalized: dict[str, int] = {}
    for end_status, periods in row.items():
        normalized_end_status = status_config.require_valid_status(end_status)
        periods = int(periods)
        if periods < 0:
            raise ValueError(
                f"Payment periods for {normalized_end_status!r} cannot be negative"
            )
        normalized[normalized_end_status] = periods

    missing_columns = status_config.valid_statuses - frozenset(normalized)
    if missing_columns:
        raise ValueError(
            "Payment matrix row is missing columns for statuses: "
            f"{sorted(missing_columns)}"
        )

    return MappingProxyType(normalized)
