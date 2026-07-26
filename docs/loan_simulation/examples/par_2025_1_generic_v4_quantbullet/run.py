"""Run PAR_2025_1 + GENERIC_v4 using QuantBullet only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from feature_builder import (
    GenericV4FeatureProvider,
    build_feature_dict,
)
from quantbullet.loan_simulation import (
    CashflowEngine,
    ConstantRecoveryLagProvider,
    ConstantSeverityProvider,
    Loan,
    LoanSimulator,
    MatrixPaymentPolicy,
    PortfolioSimulator,
    StatusConfig,
    write_simulation_workbook,
)
from quantbullet.loan_simulation.adapters import (
    build_softmax_transition_model,
    parse_rollrate_coefficients,
)


EXAMPLE_DIR = Path(__file__).resolve().parent
COEFFICIENT_DIR = EXAMPLE_DIR / "input" / "coef" / "GENERIC_v4"
LOANS_PATH = EXAMPLE_DIR / "input" / "loans_prepped_sample.json"
OUTPUT_PATH = EXAMPLE_DIR / "quantbullet_cashflows.xlsx"
HORIZON = 12
N_PATHS = 100
SEED = 20260725
START_DATE = "2025-06"

FROM_STATUSES = ["C", "D1M", "D2M", "D3M", "D4M"]
STATUS_TO_ROLL = {
    "C": ["C", "D1M", "D2M", "D3M", "D4M", "PIF", "LIQ"],
    "D1M": ["D1M", "C", "D2M", "D3M", "D4M", "PIF", "LIQ"],
    "D2M": ["D2M", "C", "D1M", "D3M", "D4M", "PIF", "LIQ"],
    "D3M": ["D3M", "C", "D1M", "D2M", "D4M", "PIF", "LIQ"],
    "D4M": ["D4M", "C", "D1M", "D2M", "D3M", "PIF", "LIQ"],
}

PAYMENT_MATRIX = {
    "C": {"C": 1, "D1M": 0, "D2M": 0, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
    "D1M": {"C": 2, "D1M": 1, "D2M": 0, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
    "D2M": {"C": 3, "D1M": 2, "D2M": 1, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
    "D3M": {"C": 4, "D1M": 3, "D2M": 2, "D3M": 1, "D4M": 0, "PIF": 0, "LIQ": 0},
    "D4M": {"C": 5, "D1M": 4, "D2M": 3, "D3M": 2, "D4M": 1, "PIF": 0, "LIQ": 0},
    "PIF": {"C": 0, "D1M": 0, "D2M": 0, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
    "LIQ": {"C": 0, "D1M": 0, "D2M": 0, "D3M": 0, "D4M": 0, "PIF": 0, "LIQ": 0},
}


class RollToOrderedTransitionModel:
    """Return probabilities in the configured roll_to order before sampling."""

    def __init__(self, base_model, status_to_roll: dict[str, list[str]]) -> None:
        self.base_model = base_model
        self.status_to_roll = status_to_roll

    def predict(
        self,
        loan,
        current_state,
        macro_features=None,
        path_features=None,
        model_features=None,
    ):
        probabilities = self.base_model.predict(
            loan,
            current_state,
            macro_features=macro_features,
            path_features=path_features,
            model_features=model_features,
        )
        roll_to = self.status_to_roll.get(current_state.status)
        if roll_to is None:
            return probabilities
        return {status: float(probabilities.get(status, 0.0)) for status in roll_to}


def build_status_config() -> StatusConfig:
    return StatusConfig(
        valid_statuses=set(STATUS_TO_ROLL) | {status for row in STATUS_TO_ROLL.values() for status in row},
        terminal_statuses={"PIF", "LIQ"},
        prepay_statuses={"PIF"},
        default_statuses={"LIQ"},
        delinquency_buckets={
            "D1M": "dq30_balance",
            "D2M": "dq60_balance",
            "D3M": "dq90_balance",
            "D4M": "dq120_balance",
        },
    )


def build_transition_model(status_config: StatusConfig):
    edge_models = {
        from_status: parse_rollrate_coefficients(
            COEFFICIENT_DIR / f"from{from_status}.txt"
        )
        for from_status in FROM_STATUSES
    }
    transition_model = build_softmax_transition_model(
        edge_models,
        feature_builder=build_feature_dict,
        status_config=status_config,
    )
    return RollToOrderedTransitionModel(transition_model, STATUS_TO_ROLL)


def build_payment_policy(status_config: StatusConfig) -> MatrixPaymentPolicy:
    return MatrixPaymentPolicy(PAYMENT_MATRIX, status_config=status_config)


def build_cashflow_engine(status_config: StatusConfig) -> CashflowEngine:
    return CashflowEngine(
        payment_policy=build_payment_policy(status_config),
        severity_provider=ConstantSeverityProvider(1.0),
        recovery_lag_provider=ConstantRecoveryLagProvider(0),
        status_config=status_config,
    )


def load_prepped_loans() -> list[dict[str, Any]]:
    with LOANS_PATH.open("r", encoding="utf-8") as file:
        loans = json.load(file)
    if not isinstance(loans, list):
        raise ValueError(f"Expected a list of loans in {LOANS_PATH}")
    return loans


def build_loans() -> list[Loan]:
    loans = []
    for feature_row in load_prepped_loans():
        balance = float(feature_row["end_bal"])
        if balance <= 0.1:
            continue
        annual_rate = float(feature_row.get("int_rate", feature_row["note_rate"]))
        loans.append(
            Loan(
                loan_id=str(feature_row["loan_id"]),
                balance=balance,
                annual_rate=annual_rate,
                term_months=int(feature_row["term"]),
                original_balance=float(feature_row["orig_bal"]),
                age_months=int(feature_row.get("loan_age", feature_row.get("age", 0))),
                status=str(feature_row["status"]),
                metadata=feature_row,
            )
        )
    return loans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QuantBullet-only demo.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status_config = build_status_config()
    loans = build_loans()
    transition_model = build_transition_model(status_config)
    cashflow_engine = build_cashflow_engine(status_config)
    loan_simulator = LoanSimulator(
        transition_model=transition_model,
        cashflow_engine=cashflow_engine,
        horizon=HORIZON,
        n_paths=N_PATHS,
        seed=SEED,
        start_date=START_DATE,
        runtime_feature_provider=GenericV4FeatureProvider.from_input_dir(
            EXAMPLE_DIR / "input"
        ),
    )
    result = PortfolioSimulator(loan_simulator).simulate(loans)
    frames = write_simulation_workbook(
        result,
        args.output,
        include_loan_cashflows=False,
        include_path_cashflows=True,
    )
    portfolio_metrics = frames["portfolio_metrics"]
    print(portfolio_metrics[["period", "cpr", "cdr", "cumulative_loss_rate"]].to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
