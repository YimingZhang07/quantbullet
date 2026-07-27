from __future__ import annotations

from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, Mapping, Sequence

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

    @classmethod
    def from_delinquency_chain(
        cls,
        delinquency_chain: Sequence[str],
        *,
        status_config: StatusConfig,
    ) -> MatrixPaymentPolicy:
        """Build the standard roll-rate payment matrix from a status ladder.

        ``delinquency_chain`` lists the performing statuses ordered by
        delinquency depth, starting with the current status. Each period one
        scheduled payment comes due, so ending at depth ``j`` after starting
        at depth ``i`` implies ``max(i + 1 - j, 0)`` payments were collected:
        staying current pays one, rolling deeper pays nothing, and a cure
        collects the catch-up installments. Transitions into terminal
        statuses collect nothing, and terminal rows are all zero.

        Every valid status must either appear in the chain or be terminal;
        build the matrix explicitly for setups with other non-terminal
        statuses.
        """
        chain = [
            status_config.require_valid_status(status)
            for status in delinquency_chain
        ]
        if not chain:
            raise ValueError("delinquency_chain must be non-empty")
        if len(set(chain)) != len(chain):
            raise ValueError(
                "delinquency_chain must not contain duplicate statuses"
            )
        terminal_in_chain = sorted(
            status for status in chain if status_config.is_terminal(status)
        )
        if terminal_in_chain:
            raise ValueError(
                "delinquency_chain must not contain terminal statuses: "
                f"{terminal_in_chain}"
            )
        uncovered = (
            status_config.valid_statuses
            - frozenset(chain)
            - status_config.terminal_statuses
        )
        if uncovered:
            raise ValueError(
                "Statuses are neither in delinquency_chain nor terminal: "
                f"{sorted(uncovered)}"
            )

        depth = {status: index for index, status in enumerate(chain)}
        statuses = chain + sorted(status_config.terminal_statuses)
        payment_periods: dict[str, dict[str, int]] = {}
        for begin_status in statuses:
            begin_depth = depth.get(begin_status)
            row: dict[str, int] = {}
            for end_status in statuses:
                end_depth = depth.get(end_status)
                if begin_depth is None or end_depth is None:
                    row[end_status] = 0
                else:
                    row[end_status] = max(begin_depth + 1 - end_depth, 0)
            payment_periods[begin_status] = row
        return cls(payment_periods, status_config=status_config)

    @property
    def payment_matrix(self) -> Mapping[str, Mapping[str, int]]:
        return self._payment_periods

    def __getstate__(self) -> dict[str, Any]:
        return {
            "payment_periods": {
                status: dict(row)
                for status, row in self._payment_periods.items()
            },
            "status_config": self.status_config,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__init__(
            state["payment_periods"],
            status_config=state["status_config"],
        )

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
