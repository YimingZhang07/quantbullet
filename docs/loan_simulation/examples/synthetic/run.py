"""Run the end-to-end synthetic loan simulation example."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_builder import SyntheticFeatureProvider
from models import (
    CHARGED_OFF,
    CURRENT,
    DELINQUENT_1,
    DELINQUENT_2,
    PREPAID,
    build_status_config,
    build_transition_model,
)
from quantbullet.loan_simulation import (
    CashflowEngine,
    ConstantRecoveryLagProvider,
    ConstantSeverityProvider,
    DataFrameMacroFeatureProvider,
    Loan,
    LoanSimulator,
    MatrixPaymentPolicy,
    PortfolioSimulator,
    StatusConfig,
    write_simulation_workbook,
)


EXAMPLE_DIR = Path(__file__).resolve().parent
INPUT_DIR = EXAMPLE_DIR / "input"
LOANS_PATH = INPUT_DIR / "loans.csv"
HPI_PATH = INPUT_DIR / "hpi.csv"
RATES_PATH = INPUT_DIR / "rates.csv"
OUTPUT_PATH = EXAMPLE_DIR / "synthetic_cashflows.xlsx"

N_PATHS = 50
SEED = 20260726
CHARGE_OFF_SEVERITY = 0.60
RECOVERY_LAG_MONTHS = 3


def load_loans(path: Path = LOANS_PATH) -> tuple[list[Loan], pd.Period]:
    frame = pd.read_csv(path)
    required_columns = {
        "loan_id",
        "origination_date",
        "original_balance",
        "balance",
        "annual_rate",
        "term_months",
        "age_months",
        "status",
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Loan input is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("Loan input must contain at least one loan")

    origination_dates = pd.to_datetime(frame["origination_date"], errors="raise")
    origination_periods = origination_dates.dt.to_period("M")
    unique_periods = origination_periods.unique()
    if len(unique_periods) != 1:
        raise ValueError("All synthetic loans must share one origination month")
    start_period = pd.Period(unique_periods[0], freq="M")

    loans = [
        Loan(
            loan_id=str(row.loan_id),
            balance=float(row.balance),
            annual_rate=float(row.annual_rate),
            term_months=int(row.term_months),
            original_balance=float(row.original_balance),
            age_months=int(row.age_months),
            status=str(row.status),
            metadata={"origination_date": str(row.origination_date)},
        )
        for row in frame.itertuples(index=False)
    ]
    return loans, start_period


def load_macro_features(
    hpi_path: Path = HPI_PATH,
    rates_path: Path = RATES_PATH,
) -> pd.DataFrame:
    hpi = _load_macro_series(hpi_path, "hpi")
    rates = _load_macro_series(rates_path, "market_rate")
    macro = hpi.merge(
        rates,
        on="date",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = macro.loc[macro["_merge"] != "both", "date"]
    if not unmatched.empty:
        dates = sorted(timestamp.date().isoformat() for timestamp in unmatched)
        raise ValueError(f"HPI and rates dates do not match: {dates}")

    return (
        macro.drop(columns="_merge")
        .sort_values("date")
        .set_index("date")[["hpi", "market_rate"]]
    )


def _load_macro_series(path: Path, value_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {"date", value_column}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError(f"{path.name} must contain at least one row")

    frame = frame.loc[:, ["date", value_column]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")
    return frame


def build_payment_policy(status_config: StatusConfig) -> MatrixPaymentPolicy:
    return MatrixPaymentPolicy.from_delinquency_chain(
        [CURRENT, DELINQUENT_1, DELINQUENT_2],
        status_config=status_config,
    )


def build_cashflow_engine(status_config: StatusConfig) -> CashflowEngine:
    return CashflowEngine(
        payment_policy=build_payment_policy(status_config),
        severity_provider=ConstantSeverityProvider(CHARGE_OFF_SEVERITY),
        recovery_lag_provider=ConstantRecoveryLagProvider(RECOVERY_LAG_MONTHS),
        status_config=status_config,
    )


def build_simulator(
    *,
    start_period: pd.Period,
    macro_features: pd.DataFrame,
    horizon: int,
    n_paths: int,
) -> LoanSimulator:
    status_config = build_status_config()
    return LoanSimulator(
        transition_model=build_transition_model(status_config),
        cashflow_engine=build_cashflow_engine(status_config),
        horizon=horizon,
        n_paths=n_paths,
        seed=SEED,
        start_date=start_period,
        macro_provider=DataFrameMacroFeatureProvider(macro_features),
        runtime_feature_provider=SyntheticFeatureProvider(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic loan simulation.")
    parser.add_argument("--loans", type=Path, default=LOANS_PATH)
    parser.add_argument("--hpi", type=Path, default=HPI_PATH)
    parser.add_argument("--rates", type=Path, default=RATES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--n-paths", type=int, default=N_PATHS)
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="Simulation months. Use 0 to run through the longest loan term.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-loans",
        type=int,
        default=0,
        help="Maximum loans to run. Use 0 or a negative value for all loans.",
    )
    parser.add_argument("--include-path-cashflows", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loans, start_period = load_loans(args.loans)
    if args.max_loans > 0:
        loans = loans[: args.max_loans]
    horizon = args.horizon if args.horizon > 0 else max(
        loan.remaining_term_months for loan in loans
    )
    simulator = build_simulator(
        start_period=start_period,
        macro_features=load_macro_features(args.hpi, args.rates),
        horizon=horizon,
        n_paths=args.n_paths,
    )
    portfolio_simulator = PortfolioSimulator(simulator)
    if args.workers > 1:
        result = portfolio_simulator.simulate_parallel(loans, workers=args.workers)
    else:
        result = portfolio_simulator.simulate(loans)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = write_simulation_workbook(
        result,
        args.output,
        include_path_cashflows=args.include_path_cashflows,
    )
    metrics = frames["portfolio_metrics"]
    print(metrics[["period", "cpr", "cdr", "cumulative_loss_rate"]].to_string(index=False))
    print(
        f"Simulated {len(loans)} loans x {args.n_paths} paths x "
        f"{horizon} months"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
