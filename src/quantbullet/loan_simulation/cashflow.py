from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .entities import Loan, LoanState, PeriodCashflow
from .payment import PaymentPolicy
from .recovery import RecoveryLagProvider, SeverityProvider
from .status import DEFAULT_STATUS_CONFIG, StatusConfig


@dataclass(frozen=True)
class RecoveryEvent:
    """Future recovery cashflow created by a default event."""

    loan_id: str
    path_id: int
    period: int
    gross_recovery: float
    recovery_cost: float = 0.0

    @property
    def net_recovery(self) -> float:
        return self.gross_recovery - self.recovery_cost


@dataclass(frozen=True)
class CashflowResult:
    """Single-period cashflow result and resulting loan state."""

    cashflow: PeriodCashflow
    next_state: LoanState
    recovery_event: RecoveryEvent | None = None


class CashflowEngine:
    """Project one fixed-rate amortizing loan period after a status transition."""

    def __init__(
        self,
        payment_policy: PaymentPolicy,
        severity_provider: SeverityProvider,
        recovery_lag_provider: RecoveryLagProvider,
        *,
        status_config: StatusConfig | None = None,
    ) -> None:
        self.status_config = status_config or DEFAULT_STATUS_CONFIG
        self.payment_policy = payment_policy
        self.severity_provider = severity_provider
        self.recovery_lag_provider = recovery_lag_provider

    def project_period(
        self,
        loan: Loan,
        begin_state: LoanState,
        end_status: str,
        *,
        path_id: int = 0,
        macro_features: Mapping[str, Any] | None = None,
        path_features: Mapping[str, Any] | None = None,
    ) -> CashflowResult:
        # Each Monte Carlo path carries its own LoanState. This method advances
        # one path by one period after the transition model has chosen end_status.
        cashflow_period = begin_state.period + 1
        begin_status = self.status_config.require_valid_status(begin_state.status)
        end_status = self.status_config.require_valid_status(end_status)

        if not begin_state.is_active(self.status_config):
            raise ValueError("begin_state must be active to project a cashflow period")

        begin_balance = begin_state.balance
        recovery_event = None

        # Prepay is a terminal payoff event: collect current-period interest and
        # all remaining principal, then close the loan on this path.
        if self.status_config.is_prepay(end_status):
            scheduled_interest, scheduled_principal, _ = _compute_scheduled_payments(
                begin_balance=begin_balance,
                monthly_rate=loan.monthly_rate,
                monthly_payment=loan.scheduled_monthly_payment,
                payment_periods=1,
            )
            interest_collected = scheduled_interest
            principal_collected = begin_balance
            end_balance = 0.0
            default_balance = 0.0
            loss = 0.0
        # Default is a terminal credit event: recognize loss now, and create a
        # future recovery instruction rather than booking delayed recovery here.
        elif self.status_config.is_default(end_status):
            scheduled_interest = 0.0
            scheduled_principal = 0.0
            interest_collected = 0.0
            principal_collected = 0.0
            default_balance = begin_balance
            severity = self.severity_provider.severity(
                loan,
                begin_state,
                end_status,
                macro_features=macro_features,
                path_features=path_features,
            )
            loss = default_balance * severity
            gross_recovery = max(default_balance - loss, 0.0)
            recovery_lag = self.recovery_lag_provider.recovery_lag_periods(
                loan,
                begin_state,
                end_status,
                macro_features=macro_features,
                path_features=path_features,
            )
            recovery_event = RecoveryEvent(
                loan_id=loan.loan_id,
                path_id=path_id,
                period=cashflow_period + recovery_lag,
                gross_recovery=gross_recovery,
            )
            end_balance = 0.0
        else:
            # Non-terminal states use the payment policy to decide how many
            # scheduled installments are collected for this status transition.
            payment_periods = self.payment_policy.payment_periods(
                loan,
                begin_state,
                end_status,
                macro_features=macro_features,
                path_features=path_features,
            )
            scheduled_interest, scheduled_principal, end_balance = (
                _compute_scheduled_payments(
                    begin_balance=begin_balance,
                    monthly_rate=loan.monthly_rate,
                    monthly_payment=loan.scheduled_monthly_payment,
                    payment_periods=payment_periods,
                )
            )
            interest_collected = scheduled_interest
            principal_collected = scheduled_principal
            default_balance = 0.0
            loss = 0.0

        # Delinquency is reporting metadata on the period cashflow. The payment
        # amount itself has already been determined by the payment policy.
        delinquency_bucket = self.status_config.delinquency_bucket(end_status)
        delinquent_balance = end_balance if delinquency_bucket else 0.0

        cashflow = PeriodCashflow(
            loan_id=loan.loan_id,
            path_id=path_id,
            period=cashflow_period,
            begin_age_months=begin_state.age_months,
            end_age_months=begin_state.age_months + 1,
            begin_balance=begin_balance,
            end_balance=end_balance,
            begin_status=begin_status,
            end_status=end_status,
            scheduled_interest=scheduled_interest,
            scheduled_principal=scheduled_principal,
            interest_collected=interest_collected,
            principal_collected=principal_collected,
            default_balance=default_balance,
            loss=loss,
            delinquency_bucket=delinquency_bucket,
            delinquent_balance=delinquent_balance,
        )
        # The next period starts from the transition outcome and aged loan.
        next_state = LoanState(
            loan_id=loan.loan_id,
            period=cashflow_period,
            age_months=begin_state.age_months + 1,
            balance=end_balance,
            status=end_status,
        )

        return CashflowResult(
            cashflow=cashflow,
            next_state=next_state,
            recovery_event=recovery_event,
        )


def _compute_scheduled_payments(
    *,
    begin_balance: float,
    monthly_rate: float,
    monthly_payment: float,
    payment_periods: int,
) -> tuple[float, float, float]:
    """Collect one or more scheduled payments from a fixed-rate loan.

    The scheduled payment amount is based on the loan's original fixed-rate
    amortization schedule. When ``payment_periods`` is greater than one, each
    payment is applied sequentially so later interest is computed on the
    reduced balance. Returns
    ``(interest_collected, principal_collected, end_balance)``.
    """
    if payment_periods < 0:
        raise ValueError("payment_periods must be non-negative")
    if begin_balance <= 0 or payment_periods == 0:
        return 0.0, 0.0, begin_balance

    balance = begin_balance
    interest_total = 0.0
    principal_total = 0.0

    for _ in range(payment_periods):
        if balance <= 0:
            break
        interest = balance * monthly_rate
        principal = min(max(monthly_payment - interest, 0.0), balance)
        balance -= principal
        interest_total += interest
        principal_total += principal

    return interest_total, principal_total, max(balance, 0.0)
