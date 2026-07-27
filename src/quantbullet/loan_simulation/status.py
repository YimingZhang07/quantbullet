from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


class LoanStatus:
    """Default status names for the first phase of the simulation framework."""

    CURRENT = "CURRENT"
    DQ30 = "DQ30"
    DQ60 = "DQ60"
    DQ90 = "DQ90"
    DEFAULTED = "DEFAULTED"
    PAID_OFF = "PAID_OFF"


@dataclass(frozen=True)
class StatusConfig:
    """Business meaning for loan status names.

    Status strings are intentionally configurable because status vocabularies
    are product- and data-specific. Callers must provide the full vocabulary
    and identify which statuses are terminal, prepay, default, or delinquency
    reporting states.
    """

    valid_statuses: set[str] | frozenset[str]
    terminal_statuses: set[str] | frozenset[str]
    prepay_statuses: set[str] | frozenset[str]
    default_statuses: set[str] | frozenset[str]
    delinquency_buckets: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_statuses = frozenset(
            normalize_status(status) for status in self.valid_statuses
        )
        terminal_statuses = frozenset(
            normalize_status(status) for status in self.terminal_statuses
        )
        prepay_statuses = frozenset(
            normalize_status(status) for status in self.prepay_statuses
        )
        default_statuses = frozenset(
            normalize_status(status) for status in self.default_statuses
        )
        delinquency_buckets = {
            normalize_status(status): str(bucket)
            for status, bucket in self.delinquency_buckets.items()
        }

        if not valid_statuses:
            raise ValueError("valid_statuses must be non-empty")

        configured_statuses = (
            terminal_statuses
            | prepay_statuses
            | default_statuses
            | frozenset(delinquency_buckets)
        )
        unknown_statuses = configured_statuses - valid_statuses
        if unknown_statuses:
            raise ValueError(
                "Status configuration contains statuses not in valid_statuses: "
                f"{sorted(unknown_statuses)}"
            )

        non_terminal = (prepay_statuses | default_statuses) - terminal_statuses
        if non_terminal:
            raise ValueError(
                "prepay_statuses and default_statuses must be terminal: "
                f"{sorted(non_terminal)}"
            )

        object.__setattr__(self, "valid_statuses", valid_statuses)
        object.__setattr__(self, "terminal_statuses", terminal_statuses)
        object.__setattr__(self, "prepay_statuses", prepay_statuses)
        object.__setattr__(self, "default_statuses", default_statuses)
        object.__setattr__(self, "delinquency_buckets", delinquency_buckets)

    def require_valid_status(self, status: str) -> str:
        normalized = normalize_status(status)
        if normalized not in self.valid_statuses:
            raise ValueError(
                f"Unknown status {normalized!r}; expected one of "
                f"{sorted(self.valid_statuses)}"
            )
        return normalized

    def is_valid_status(self, status: str) -> bool:
        return normalize_status(status) in self.valid_statuses

    def is_terminal(self, status: str) -> bool:
        return self.require_valid_status(status) in self.terminal_statuses

    def is_prepay(self, status: str) -> bool:
        return self.require_valid_status(status) in self.prepay_statuses

    def is_default(self, status: str) -> bool:
        return self.require_valid_status(status) in self.default_statuses

    def delinquency_bucket(self, status: str) -> str | None:
        return self.delinquency_buckets.get(self.require_valid_status(status))

    def is_delinquent(self, status: str) -> bool:
        return self.delinquency_bucket(status) is not None


def normalize_status(status: str) -> str:
    normalized = str(status)
    if not normalized:
        raise ValueError("status must be non-empty")
    return normalized


DEFAULT_STATUS_CONFIG = StatusConfig(
    valid_statuses=frozenset(
        {
            LoanStatus.CURRENT,
            LoanStatus.DQ30,
            LoanStatus.DQ60,
            LoanStatus.DQ90,
            LoanStatus.DEFAULTED,
            LoanStatus.PAID_OFF,
        }
    ),
    terminal_statuses=frozenset({LoanStatus.DEFAULTED, LoanStatus.PAID_OFF}),
    prepay_statuses=frozenset({LoanStatus.PAID_OFF}),
    default_statuses=frozenset({LoanStatus.DEFAULTED}),
    delinquency_buckets={
        LoanStatus.DQ30: "dq30_balance",
        LoanStatus.DQ60: "dq60_balance",
        LoanStatus.DQ90: "dq90_balance",
    },
)
