from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .status import DEFAULT_STATUS_CONFIG, LoanStatus, StatusConfig, normalize_status


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
        status = normalize_status(self.status)
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
        self.status = normalize_status(self.status)
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

