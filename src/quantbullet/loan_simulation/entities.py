from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


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

    Status strings are intentionally configurable. The defaults cover a simple
    amortizing-loan setup, while custom use cases can add statuses such as
    LIQ, SOLD, REFI, or CHARGED_OFF without changing the engine.
    """

    terminal_statuses: set[str] | frozenset[str] = field(
        default_factory=lambda: frozenset(
            {LoanStatus.DEFAULTED, LoanStatus.PAID_OFF}
        )
    )
    prepay_statuses: set[str] | frozenset[str] = field(
        default_factory=lambda: frozenset({LoanStatus.PAID_OFF})
    )
    default_statuses: set[str] | frozenset[str] = field(
        default_factory=lambda: frozenset({LoanStatus.DEFAULTED})
    )
    delinquency_buckets: Mapping[str, str] = field(
        default_factory=lambda: {
            LoanStatus.DQ30: "dq30_balance",
            LoanStatus.DQ60: "dq60_balance",
            LoanStatus.DQ90: "dq90_balance",
        }
    )

    def __post_init__(self) -> None:
        terminal_statuses = frozenset(
            _normalize_status(status) for status in self.terminal_statuses
        )
        prepay_statuses = frozenset(
            _normalize_status(status) for status in self.prepay_statuses
        )
        default_statuses = frozenset(
            _normalize_status(status) for status in self.default_statuses
        )
        delinquency_buckets = {
            _normalize_status(status): str(bucket)
            for status, bucket in self.delinquency_buckets.items()
        }

        non_terminal = (prepay_statuses | default_statuses) - terminal_statuses
        if non_terminal:
            raise ValueError(
                "prepay_statuses and default_statuses must be terminal: "
                f"{sorted(non_terminal)}"
            )

        object.__setattr__(self, "terminal_statuses", terminal_statuses)
        object.__setattr__(self, "prepay_statuses", prepay_statuses)
        object.__setattr__(self, "default_statuses", default_statuses)
        object.__setattr__(self, "delinquency_buckets", delinquency_buckets)

    def is_terminal(self, status: str) -> bool:
        return _normalize_status(status) in self.terminal_statuses

    def is_prepay(self, status: str) -> bool:
        return _normalize_status(status) in self.prepay_statuses

    def is_default(self, status: str) -> bool:
        return _normalize_status(status) in self.default_statuses

    def delinquency_bucket(self, status: str) -> str | None:
        return self.delinquency_buckets.get(_normalize_status(status))

    def is_delinquent(self, status: str) -> bool:
        return self.delinquency_bucket(status) is not None


@dataclass(frozen=True)
class Loan:
    """Fixed-rate amortizing loan input used by the simulator."""

    loan_id: str
    balance: float
    annual_rate: float
    term_months: int
    age_months: int = 0
    status: str = LoanStatus.CURRENT
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = _normalize_status(self.status)
        object.__setattr__(self, "status", status)

        if not self.loan_id:
            raise ValueError("loan_id must be non-empty")
        if self.balance < 0:
            raise ValueError("balance must be non-negative")
        if self.annual_rate < 0:
            raise ValueError("annual_rate must be non-negative")
        if self.term_months <= 0:
            raise ValueError("term_months must be positive")
        if self.age_months < 0:
            raise ValueError("age_months must be non-negative")
        if self.age_months > self.term_months:
            raise ValueError("age_months cannot exceed term_months")

    @property
    def monthly_rate(self) -> float:
        return self.annual_rate / 12.0

    @property
    def remaining_term_months(self) -> int:
        return max(self.term_months - self.age_months, 0)

    def initial_state(self) -> LoanState:
        return LoanState(
            loan_id=self.loan_id,
            period=0,
            age_months=self.age_months,
            balance=self.balance,
            status=self.status,
        )


@dataclass
class LoanState:
    """Mutable point-in-time loan state used inside a simulation path."""

    loan_id: str
    period: int
    age_months: int
    balance: float
    status: str = LoanStatus.CURRENT

    def __post_init__(self) -> None:
        self.status = _normalize_status(self.status)
        if self.period < 0:
            raise ValueError("period must be non-negative")
        if self.age_months < 0:
            raise ValueError("age_months must be non-negative")
        if self.balance < 0:
            raise ValueError("balance must be non-negative")

    def is_active(self, status_config: StatusConfig | None = None) -> bool:
        config = status_config or DEFAULT_STATUS_CONFIG
        return self.balance > 0 and not config.is_terminal(self.status)


@dataclass(frozen=True)
class PeriodCashflow:
    """Cashflow and balance record for one projected loan period."""

    loan_id: str
    path_id: int
    period: int
    age_months: int
    begin_balance: float
    end_balance: float
    begin_status: str
    end_status: str
    scheduled_interest: float = 0.0
    scheduled_principal: float = 0.0
    interest_collected: float = 0.0
    principal_collected: float = 0.0
    default_balance: float = 0.0
    loss: float = 0.0
    gross_recovery: float = 0.0
    recovery_cost: float = 0.0
    net_recovery: float = 0.0
    delinquency_bucket: str | None = None
    delinquent_balance: float = 0.0

    @property
    def prepayment_amount(self) -> float:
        return max(self.principal_collected - self.scheduled_principal, 0.0)

    @property
    def total_cashflow(self) -> float:
        return (
            self.interest_collected
            + self.principal_collected
            + self.net_recovery
        )


def _normalize_status(status: str) -> str:
    normalized = str(status)
    if not normalized:
        raise ValueError("status must be non-empty")
    return normalized


DEFAULT_STATUS_CONFIG = StatusConfig()
