from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from multiprocessing import Pool
from typing import Any

import pandas as pd

from .cashflow import CashflowEngine, RecoveryEvent
from .entities import Loan, LoanState, PeriodCashflow
from .macro import MacroFeatureProvider
from .path_features import PathFeatureTracker
from .runtime_features import EmptyRuntimeFeatureProvider, RuntimeFeatureProvider
from .transition import TransitionModel, sample_next_status

_PARALLEL_LOAN_SIMULATOR: Any = None


def _init_parallel_worker(loan_simulator: LoanSimulator) -> None:
    global _PARALLEL_LOAN_SIMULATOR
    _PARALLEL_LOAN_SIMULATOR = loan_simulator


def _simulate_loan_parallel(loan: Loan) -> LoanSimulationResult:
    if _PARALLEL_LOAN_SIMULATOR is None:
        raise RuntimeError("parallel loan simulator was not initialized")
    return _PARALLEL_LOAN_SIMULATOR.simulate_loan(loan)


def _simulate_loan_totals_parallel(
    loan: Loan,
) -> dict[int, tuple[float, ...]]:
    if _PARALLEL_LOAN_SIMULATOR is None:
        raise RuntimeError("parallel loan simulator was not initialized")
    return _PARALLEL_LOAN_SIMULATOR.simulate_loan_totals(loan)


AGGREGATE_CASHFLOW_FIELDS = (
    "begin_balance",
    "end_balance",
    "scheduled_interest",
    "scheduled_principal",
    "interest_collected",
    "principal_collected",
    "default_balance",
    "loss",
    "gross_recovery",
    "recovery_cost",
    "net_recovery",
    "delinquent_balance",
    "prepayment_amount",
    "total_cashflow",
)


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
        frames = []
        for result in self.loan_results:
            frame = result.to_frame()
            if frame.empty:
                continue
            numeric_cols = [
                column
                for column in _numeric_cashflow_columns(frame)
                if column != "original_balance"
            ]
            grouped = (
                frame.groupby(
                    ["loan_id", "period", "period_date"],
                    as_index=False,
                )[numeric_cols]
                .sum()
                .sort_values(["loan_id", "period"])
                .reset_index(drop=True)
            )
            grouped[numeric_cols] = grouped[numeric_cols].div(result.n_paths)
            grouped["original_balance"] = result.loan.original_balance
            frames.append(grouped)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def portfolio_cashflows(self) -> pd.DataFrame:
        frame = self.loan_cashflows()
        if frame.empty:
            return frame
        numeric_cols = _numeric_cashflow_columns(frame)
        portfolio = (
            frame.groupby(["period", "period_date"], as_index=False)[numeric_cols]
            .sum()
            .sort_values("period")
            .reset_index(drop=True)
        )
        portfolio["original_balance"] = sum(
            float(result.loan.original_balance) for result in self.loan_results
        )
        return portfolio


@dataclass(frozen=True)
class PortfolioAggregateResult:
    """Compact portfolio totals for runs that do not retain path rows."""

    period_totals: dict[int, tuple[float, ...]]
    start_period: pd.Period
    original_balance: float

    def portfolio_cashflows(self) -> pd.DataFrame:
        rows = []
        for period in sorted(self.period_totals):
            values = self.period_totals[period]
            row = {
                "period": period,
                "period_date": str(self.start_period + period),
            }
            row.update(
                {
                    field: value
                    for field, value in zip(
                        AGGREGATE_CASHFLOW_FIELDS,
                        values,
                    )
                }
            )
            row["original_balance"] = self.original_balance
            rows.append(row)
        return pd.DataFrame(rows)


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
        runtime_feature_provider: RuntimeFeatureProvider | None = None,
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
        self.runtime_feature_provider = (
            runtime_feature_provider or EmptyRuntimeFeatureProvider()
        )

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

    def simulate_loan_totals(self, loan: Loan) -> dict[int, tuple[float, ...]]:
        totals: dict[int, list[float]] = {}
        for path_id in range(self.n_paths):
            rng = random.Random(_stable_seed(self.seed, loan.loan_id, path_id))
            for cashflow in self._simulate_path(loan, path_id=path_id, rng=rng):
                period_totals = totals.setdefault(
                    cashflow.period,
                    [0.0] * len(AGGREGATE_CASHFLOW_FIELDS),
                )
                for index, value in enumerate(_aggregate_cashflow_values(cashflow)):
                    period_totals[index] += value
        return {
            period: tuple(value / self.n_paths for value in values)
            for period, values in totals.items()
        }

    def _simulate_path(
        self,
        loan: Loan,
        *,
        path_id: int,
        rng: random.Random,
    ) -> list[PeriodCashflow]:
        state = loan.initial_state()
        path_feature_tracker = PathFeatureTracker()
        runtime_feature_state = self.runtime_feature_provider.initialize_path_state(
            loan,
            self.start_period,
        )
        pending_recoveries: dict[int, list[RecoveryEvent]] = {}
        cashflows: list[PeriodCashflow] = []

        period = 1
        while period <= self.horizon or pending_recoveries:
            due_recoveries = pending_recoveries.pop(period, [])

            if period <= self.horizon and state.is_active(
                self.cashflow_engine.status_config
            ):
                macro_features = self._macro_features_for_period(period)
                path_features = path_feature_tracker.features()
                period_date = self.start_period + period
                self.runtime_feature_provider.prepare_period_state(
                    loan=loan,
                    current_state=state,
                    period_date=period_date,
                    macro_features=macro_features,
                    path_features=path_features,
                    feature_state=runtime_feature_state,
                )
                model_features = (
                    self.runtime_feature_provider.model_features_for_period(
                        loan=loan,
                        current_state=state,
                        period_date=period_date,
                        macro_features=macro_features,
                        path_features=path_features,
                        feature_state=runtime_feature_state,
                    )
                )
                probabilities = self.transition_model.predict(
                    loan,
                    state,
                    macro_features=macro_features,
                    path_features=path_features,
                    model_features=model_features,
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
                path_feature_tracker.update(
                    cashflow, self.cashflow_engine.status_config
                )
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

    def simulate_parallel(
        self,
        loans: Iterable[Loan],
        *,
        workers: int,
        chunksize: int | None = None,
    ) -> PortfolioSimulationResult:
        if workers <= 1:
            return self.simulate(loans)

        loan_list = list(loans)
        if not loan_list:
            return PortfolioSimulationResult([])

        with Pool(
            processes=workers,
            initializer=_init_parallel_worker,
            initargs=(self.loan_simulator,),
        ) as pool:
            loan_results = pool.map(
                _simulate_loan_parallel,
                loan_list,
                chunksize=chunksize,
            )
        return PortfolioSimulationResult(loan_results)

    def simulate_parallel_aggregate(
        self,
        loans: Iterable[Loan],
        *,
        workers: int,
        chunksize: int | None = None,
    ) -> PortfolioAggregateResult:
        loan_list = list(loans)
        portfolio_totals: dict[int, list[float]] = {}

        if workers <= 1:
            for loan in loan_list:
                _merge_period_totals(
                    portfolio_totals,
                    self.loan_simulator.simulate_loan_totals(loan),
                )
        elif loan_list:
            effective_chunksize = chunksize or max(
                1,
                len(loan_list) // (workers * 4),
            )
            with Pool(
                processes=workers,
                initializer=_init_parallel_worker,
                initargs=(self.loan_simulator,),
            ) as pool:
                for loan_totals in pool.imap(
                    _simulate_loan_totals_parallel,
                    loan_list,
                    chunksize=effective_chunksize,
                ):
                    _merge_period_totals(portfolio_totals, loan_totals)

        return PortfolioAggregateResult(
            period_totals={
                period: tuple(values) for period, values in portfolio_totals.items()
            },
            start_period=self.loan_simulator.start_period,
            original_balance=sum(float(loan.original_balance) for loan in loan_list),
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


def _aggregate_cashflow_values(cashflow: PeriodCashflow) -> tuple[float, ...]:
    return (
        cashflow.begin_balance,
        cashflow.end_balance,
        cashflow.scheduled_interest,
        cashflow.scheduled_principal,
        cashflow.interest_collected,
        cashflow.principal_collected,
        cashflow.default_balance,
        cashflow.loss,
        cashflow.gross_recovery,
        cashflow.recovery_cost,
        cashflow.net_recovery,
        cashflow.delinquent_balance,
        cashflow.prepayment_amount,
        cashflow.total_cashflow,
    )


def _merge_period_totals(
    portfolio_totals: dict[int, list[float]],
    loan_totals: dict[int, tuple[float, ...]],
) -> None:
    for period, values in loan_totals.items():
        totals = portfolio_totals.setdefault(period, [0.0] * len(values))
        for index, value in enumerate(values):
            totals[index] += value


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
