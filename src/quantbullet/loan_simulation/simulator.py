from __future__ import annotations

import random
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping

import pandas as pd

from .cashflow import CashflowEngine, RecoveryEvent
from .entities import Loan, LoanState, PeriodCashflow
from .macro import MacroFeatureProvider
from .path_features import PathFeatureTracker
from .transition import TransitionModel, sample_next_status


@dataclass(frozen=True)
class LoanSimulationResult:
    """Path-level simulation result for one loan."""

    loan: Loan
    cashflows: list[PeriodCashflow]
    start_period: pd.Period
    n_paths: int = 1

    def to_frame(self) -> pd.DataFrame:
        return _cashflows_to_frame(
            self.cashflows,
            loan=self.loan,
            start_period=self.start_period,
        )


@dataclass(frozen=True)
class PortfolioSimulationResult:
    """Simulation result for a collection of loans."""

    loan_results: list[LoanSimulationResult]

    def path_cashflows(self) -> pd.DataFrame:
        frames = [result.to_frame() for result in self.loan_results]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def loan_cashflows(self) -> pd.DataFrame:
        frame = self.path_cashflows()
        if frame.empty:
            return frame
        numeric_cols = [
            column
            for column in _numeric_cashflow_columns(frame)
            if column != "original_balance"
        ]
        loan_paths = {
            result.loan.loan_id: result.n_paths
            for result in self.loan_results
        }
        original_balances = {
            result.loan.loan_id: result.loan.original_balance
            for result in self.loan_results
        }
        grouped = (
            frame.groupby(["loan_id", "period", "period_date"], as_index=False)[numeric_cols]
            .sum()
            .sort_values(["loan_id", "period"])
            .reset_index(drop=True)
        )
        path_counts = grouped["loan_id"].map(loan_paths)
        grouped[numeric_cols] = grouped[numeric_cols].div(path_counts, axis=0)
        grouped["original_balance"] = grouped["loan_id"].map(original_balances)
        return grouped

    def portfolio_cashflows(self) -> pd.DataFrame:
        frame = self.loan_cashflows()
        if frame.empty:
            return frame
        numeric_cols = _numeric_cashflow_columns(frame)
        return (
            frame.groupby(["period", "period_date"], as_index=False)[numeric_cols]
            .sum()
            .sort_values("period")
            .reset_index(drop=True)
        )


class LoanSimulator:
    """Sequential Monte Carlo simulator for loan-level cashflows."""

    def __init__(
        self,
        transition_model: TransitionModel,
        cashflow_engine: CashflowEngine,
        *,
        horizon: int,
        n_paths: int = 1,
        seed: int = 42,
        start_date: Any,
        frequency: str = "M",
        macro_provider: MacroFeatureProvider | None = None,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        self.transition_model = transition_model
        self.cashflow_engine = cashflow_engine
        self.horizon = horizon
        self.n_paths = n_paths
        self.seed = seed
        self.start_period = pd.Period(start_date, freq=frequency)
        self.frequency = frequency
        self.macro_provider = macro_provider

    def simulate_loan(self, loan: Loan) -> LoanSimulationResult:
        cashflows: list[PeriodCashflow] = []
        for path_id in range(self.n_paths):
            rng = random.Random(_stable_seed(self.seed, loan.loan_id, path_id))
            cashflows.extend(self._simulate_path(loan, path_id=path_id, rng=rng))
        return LoanSimulationResult(
            loan=loan,
            cashflows=cashflows,
            start_period=self.start_period,
            n_paths=self.n_paths,
        )

    def _simulate_path(
        self,
        loan: Loan,
        *,
        path_id: int,
        rng: random.Random,
    ) -> list[PeriodCashflow]:
        state = loan.initial_state()
        path_feature_tracker = PathFeatureTracker()
        pending_recoveries: dict[int, list[RecoveryEvent]] = {}
        cashflows: list[PeriodCashflow] = []

        period = 1
        while period <= self.horizon or pending_recoveries:
            due_recoveries = pending_recoveries.pop(period, [])

            if period <= self.horizon and state.is_active(self.cashflow_engine.status_config):
                macro_features = self._macro_features_for_period(period)
                path_features = path_feature_tracker.features()
                probabilities = self.transition_model.predict(
                    loan,
                    state,
                    macro_features=macro_features,
                    path_features=path_features,
                )
                end_status = sample_next_status(probabilities, rng)
                result = self.cashflow_engine.project_period(
                    loan,
                    state,
                    end_status,
                    path_id=path_id,
                    macro_features=macro_features,
                    path_features=path_features,
                )

                cashflow = result.cashflow
                if result.recovery_event is not None:
                    if result.recovery_event.period == period:
                        due_recoveries.append(result.recovery_event)
                    else:
                        pending_recoveries.setdefault(
                            result.recovery_event.period,
                            [],
                        ).append(result.recovery_event)

                if due_recoveries:
                    cashflow = _add_recoveries(cashflow, due_recoveries)
                cashflows.append(cashflow)
                path_feature_tracker.update(cashflow, self.cashflow_engine.status_config)
                state = result.next_state
            elif due_recoveries:
                cashflows.append(
                    _recovery_only_cashflow(
                        loan=loan,
                        state=state,
                        path_id=path_id,
                        period=period,
                        recoveries=due_recoveries,
                    )
                )
            elif pending_recoveries:
                period = min(pending_recoveries)
                continue
            elif not pending_recoveries:
                break

            period += 1

        return cashflows

    def _macro_features_for_period(self, period: int) -> Mapping[str, Any]:
        if self.macro_provider is None:
            return {}
        return self.macro_provider.features_for_date(self.start_period + period)


class PortfolioSimulator:
    """Sequential portfolio wrapper around ``LoanSimulator``."""

    def __init__(self, loan_simulator: LoanSimulator) -> None:
        self.loan_simulator = loan_simulator

    def simulate(self, loans: Iterable[Loan]) -> PortfolioSimulationResult:
        return PortfolioSimulationResult(
            [self.loan_simulator.simulate_loan(loan) for loan in loans]
        )


def _cashflows_to_frame(
    cashflows: list[PeriodCashflow],
    *,
    loan: Loan,
    start_period: pd.Period,
) -> pd.DataFrame:
    rows = []
    for cashflow in cashflows:
        row = asdict(cashflow)
        row["original_balance"] = loan.original_balance
        row["period_date"] = str(start_period + cashflow.period)
        row["prepayment_amount"] = cashflow.prepayment_amount
        row["total_cashflow"] = cashflow.total_cashflow
        rows.append(row)
    return pd.DataFrame(rows)


def _numeric_cashflow_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {
        "loan_id",
        "path_id",
        "period",
        "period_date",
        "begin_age_months",
        "end_age_months",
        "begin_status",
        "end_status",
        "delinquency_bucket",
    }
    return [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in exclude
    ]


def _add_recoveries(
    cashflow: PeriodCashflow,
    recoveries: list[RecoveryEvent],
) -> PeriodCashflow:
    gross_recovery = cashflow.gross_recovery + sum(
        event.gross_recovery for event in recoveries
    )
    recovery_cost = cashflow.recovery_cost + sum(
        event.recovery_cost for event in recoveries
    )
    net_recovery = cashflow.net_recovery + sum(
        event.net_recovery for event in recoveries
    )
    return replace(
        cashflow,
        gross_recovery=gross_recovery,
        recovery_cost=recovery_cost,
        net_recovery=net_recovery,
    )


def _recovery_only_cashflow(
    *,
    loan: Loan,
    state: LoanState,
    path_id: int,
    period: int,
    recoveries: list[RecoveryEvent],
) -> PeriodCashflow:
    age_months = state.age_months + max(period - state.period, 0)
    return _add_recoveries(
        PeriodCashflow(
            loan_id=loan.loan_id,
            path_id=path_id,
            period=period,
            begin_age_months=age_months,
            end_age_months=age_months,
            begin_balance=state.balance,
            end_balance=state.balance,
            begin_status=state.status,
            end_status=state.status,
        ),
        recoveries,
    )


def _stable_seed(seed: int, loan_id: str, path_id: int) -> int:
    value = f"{seed}|{loan_id}|{path_id}"
    hash_value = 1469598103934665603
    for character in value:
        hash_value ^= ord(character)
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(hash_value % (2**32))
