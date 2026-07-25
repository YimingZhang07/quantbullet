"""Small Monte Carlo cashflow benchmark for PAR_2025_1 + GENERIC_v4."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from feature_builder import init_runtime_feature_state, step_runtime_features
from quantbullet.loan_simulation import (
    CashflowEngine,
    ConstantRecoveryLagProvider,
    ConstantSeverityProvider,
    Loan,
    LoanSimulator,
    MatrixPaymentPolicy,
    PortfolioSimulator,
    StatusConfig,
    compute_period_metrics,
)
from quantbullet.loan_simulation.adapters import (
    build_softmax_transition_model,
    parse_rollrate_coefficients,
)


EXAMPLE_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_prepped_loans(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        loans = json.load(file)
    if not isinstance(loans, list):
        raise ValueError(f"Expected a list of loans in {path}")
    return loans


def build_status_config(config: dict[str, Any]) -> StatusConfig:
    return StatusConfig(
        valid_statuses=_status_universe(config["status_to_roll"]),
        terminal_statuses=set(config["terminal_statuses"]),
        prepay_statuses=set(config["prepay_statuses"]),
        default_statuses=set(config["default_statuses"]),
        delinquency_buckets=dict(config["delinquency_buckets"]),
    )


def _status_universe(status_to_roll: dict[str, list[str]]) -> set[str]:
    statuses = set(status_to_roll)
    for row_statuses in status_to_roll.values():
        statuses.update(row_statuses)
    return statuses


def build_feature_builder(config: dict[str, Any]):
    def build_feature_dict(context):
        state = init_runtime_feature_state(dict(context.loan.metadata), config)
        for period in range(1, context.current_state.period + 1):
            state = step_runtime_features(state, next_period=period, config=config)
        state["status"] = context.current_state.status
        state["end_bal"] = context.current_state.balance
        return state

    return build_feature_dict


def load_edge_models(
    coefficient_root: Path,
    *,
    from_statuses: list[str],
) -> dict[str, dict[str, Any]]:
    return {
        from_status: parse_rollrate_coefficients(
            coefficient_root / f"from{from_status}.txt"
        )
        for from_status in from_statuses
    }


def load_payment_matrix(path: Path, status_config: StatusConfig) -> dict[str, dict[str, int]]:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    matrix = {}
    for begin_status in status_config.valid_statuses:
        matrix[begin_status] = {}
        for end_status in status_config.valid_statuses:
            value = frame.loc[begin_status, end_status] if begin_status in frame.index else 0
            periods = int(value) if end_status in frame.columns else 0
            if status_config.is_terminal(end_status):
                periods = 0
            matrix[begin_status][end_status] = periods
    return matrix


class RollToOrderedTransitionModel:
    """Return probabilities in roll-rate's configured roll_to order."""

    def __init__(self, base_model, status_to_roll: dict[str, list[str]]) -> None:
        self.base_model = base_model
        self.status_to_roll = status_to_roll

    def predict(
        self,
        loan,
        current_state,
        macro_features=None,
        path_features=None,
    ):
        probabilities = self.base_model.predict(
            loan,
            current_state,
            macro_features=macro_features,
            path_features=path_features,
        )
        roll_to = self.status_to_roll.get(current_state.status)
        if roll_to is None:
            return probabilities
        return {status: float(probabilities.get(status, 0.0)) for status in roll_to}


def to_quantbullet_loans(prepped_loans: list[dict[str, Any]]) -> list[Loan]:
    loans = []
    for feature_row in prepped_loans:
        balance = float(feature_row["end_bal"])
        if balance <= 0.1:
            continue
        annual_rate = float(feature_row.get("int_rate", feature_row["note_rate"]))
        term_months = int(feature_row["term"])
        loans.append(
            Loan(
                loan_id=str(feature_row["loan_id"]),
                balance=balance,
                annual_rate=annual_rate,
                term_months=term_months,
                original_balance=float(feature_row["orig_bal"]),
                age_months=int(feature_row.get("loan_age", feature_row.get("age", 0))),
                status=str(feature_row["status"]),
                scheduled_payment=_rollrate_level_payment(
                    balance,
                    annual_rate,
                    term_months,
                ),
                metadata=feature_row,
            )
        )
    return loans


def _rollrate_level_payment(
    balance: float,
    annual_rate: float,
    term_months: int,
) -> float:
    monthly_rate = annual_rate / 12.0 if annual_rate < 1.0 else annual_rate / 1200.0
    if monthly_rate == 0:
        return balance / term_months
    growth = (1.0 + monthly_rate) ** term_months
    return balance * monthly_rate * growth / (growth - 1.0)


def run_quantbullet_cashflows(
    loans: list[Loan],
    *,
    config: dict[str, Any],
    coefficient_root: Path,
    payment_matrix_path: Path,
    horizon: int,
    n_paths: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    status_config = build_status_config(config)
    edge_models = load_edge_models(
        coefficient_root,
        from_statuses=list(config["from_statuses"]),
    )
    transition_model = build_softmax_transition_model(
        edge_models,
        feature_builder=build_feature_builder(config),
        status_config=status_config,
    )
    transition_model = RollToOrderedTransitionModel(
        transition_model,
        config["status_to_roll"],
    )
    cashflow_engine = CashflowEngine(
        payment_policy=MatrixPaymentPolicy(
            load_payment_matrix(payment_matrix_path, status_config),
            status_config=status_config,
        ),
        severity_provider=ConstantSeverityProvider(1.0),
        recovery_lag_provider=ConstantRecoveryLagProvider(0),
        status_config=status_config,
    )
    simulator = LoanSimulator(
        transition_model=transition_model,
        cashflow_engine=cashflow_engine,
        horizon=horizon,
        n_paths=n_paths,
        seed=seed,
        start_date="2025-06",
    )
    result = PortfolioSimulator(simulator).simulate(loans)
    return result.path_cashflows(), result.portfolio_cashflows()


def run_rollrate_cashflows(
    prepped_loans: list[dict[str, Any]],
    *,
    roll_rate_root: Path,
    config: dict[str, Any],
    coefficient_root: Path,
    payment_matrix_path: Path,
    horizon: int,
    n_paths: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _add_rollrate_python_path(roll_rate_root)
    from simengine.data_prep import (  # noqa: PLC0415
        _get_registry,
        build_all_models,
        classify_model_terms,
        read_coef_file,
    )
    from simengine.runner import CF_COL, compute_metrics, run_cf_one  # noqa: PLC0415
    from simengine.runner import _build_transition_layout  # noqa: PLC0415

    status_config = build_status_config(config)
    payment_matrix = load_payment_matrix(payment_matrix_path, status_config)
    pmt_matrix_to_from = {
        to_status: {
            from_status: payment_matrix[from_status][to_status]
            for from_status in payment_matrix
        }
        for to_status in status_config.valid_statuses
    }
    coef_by_from = {
        from_status: read_coef_file(str(coefficient_root / f"from{from_status}.txt"))
        for from_status in config["from_statuses"]
    }
    models = build_all_models(coef_by_from)
    classify_model_terms(models, _get_registry().time_varying_names())
    dm = SimpleNamespace(
        n_per=horizon,
        models=models,
        status_to_roll=config["status_to_roll"],
        terminal_statuses=set(config["terminal_statuses"]),
        pmt_matrix=pmt_matrix_to_from,
        liq_severity=1.0,
        dq_buckets={
            status: (bucket.replace("_balance", ""), bucket.replace("_balance", "_bal"))
            for status, bucket in config["delinquency_buckets"].items()
        },
        clean_status_dict={
            status: status.split(".")[0]
            for statuses in config["status_to_roll"].values()
            for status in statuses
        },
        prob_layout={},
        dial_data={},
    )
    dm._transition_layout = _build_transition_layout(dm)

    portfolio_cf = [[0.0] * len(CF_COL) for _ in range(horizon)]
    eligible_loans = [
        dict(loan)
        for loan in prepped_loans
        if float(loan.get("end_bal", 0.0)) > 0.1
    ]
    for loan in eligible_loans:
        loan_cf = [[0.0] * len(CF_COL) for _ in range(horizon)]
        for path_id in range(n_paths):
            path_cf, _ = run_cf_one(
                loan,
                dm,
                dup=1,
                seed=_stable_seed(seed, str(loan["loan_id"]), path_id),
            )
            for period in range(horizon):
                for col_idx in range(len(CF_COL)):
                    loan_cf[period][col_idx] += path_cf[period][col_idx] / n_paths
        for period in range(horizon):
            for col_idx in range(len(CF_COL)):
                portfolio_cf[period][col_idx] += loan_cf[period][col_idx]

    frame = pd.DataFrame(portfolio_cf, columns=CF_COL)
    frame.insert(0, "period", range(1, horizon + 1))
    total_orig_bal = sum(float(loan["orig_bal"]) for loan in eligible_loans)
    metrics = pd.DataFrame(compute_metrics(portfolio_cf, total_orig_bal))
    return frame, metrics


def _stable_seed(seed: int, loan_id: str, path_id: int) -> int:
    value = f"{seed}|{loan_id}|{path_id}"
    hash_value = 1469598103934665603
    for character in value:
        hash_value ^= ord(character)
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(hash_value % (2**32))


def compare_cashflows(
    quantbullet_cashflows: pd.DataFrame,
    rollrate_cashflows: pd.DataFrame,
) -> pd.DataFrame:
    quantbullet = quantbullet_cashflows[
        ["period", "begin_balance", "end_balance", "interest_collected", "principal_collected", "loss"]
    ].rename(
        columns={
            "begin_balance": "quantbullet_begin_balance",
            "end_balance": "quantbullet_end_balance",
            "interest_collected": "quantbullet_interest",
            "principal_collected": "quantbullet_principal",
            "loss": "quantbullet_loss",
        }
    )
    rollrate = rollrate_cashflows[
        ["period", "begin_bal", "end_bal", "int_pmt", "prin_pmt", "loss"]
    ].rename(
        columns={
            "begin_bal": "rollrate_begin_balance",
            "end_bal": "rollrate_end_balance",
            "int_pmt": "rollrate_interest",
            "prin_pmt": "rollrate_principal",
            "loss": "rollrate_loss",
        }
    )
    diff = quantbullet.merge(rollrate, on="period", how="outer").sort_values("period")
    for field in ["begin_balance", "end_balance", "interest", "principal", "loss"]:
        diff[f"{field}_diff"] = diff[f"quantbullet_{field}"] - diff[f"rollrate_{field}"]
    return diff


def rollrate_style_cpr_from_quantbullet_paths(
    path_cashflows: pd.DataFrame,
    loans: list[Loan],
) -> pd.DataFrame:
    loan_terms = pd.DataFrame(
        [
            {
                "loan_id": loan.loan_id,
                "monthly_rate": loan.monthly_rate,
                "scheduled_payment": loan.scheduled_monthly_payment,
            }
            for loan in loans
        ]
    )
    frame = path_cashflows.merge(loan_terms, on="loan_id", how="left")
    frame["rollrate_scheduled_principal"] = (
        frame["scheduled_payment"] - frame["begin_balance"] * frame["monthly_rate"]
    ).clip(lower=0.0)
    frame["rollrate_scheduled_principal"] = frame[
        ["rollrate_scheduled_principal", "begin_balance"]
    ].min(axis=1)
    frame["payoff_event_balance"] = frame["begin_balance"].where(
        frame["end_status"].eq("PIF"),
        0.0,
    )
    grouped = (
        frame.groupby("period", as_index=False)[
            ["begin_balance", "rollrate_scheduled_principal", "payoff_event_balance"]
        ]
        .sum()
        .sort_values("period")
        .reset_index(drop=True)
    )
    denominator = (
        grouped["begin_balance"] - grouped["rollrate_scheduled_principal"]
    ).clip(lower=0.0)
    smm = (
        grouped["payoff_event_balance"] / denominator.replace(0, float("nan"))
    ).fillna(0.0)
    grouped["rollrate_style_cpr"] = 1.0 - (1.0 - smm.clip(upper=1.0)) ** 12
    return grouped[["period", "rollrate_style_cpr"]]


def compare_metrics(
    quantbullet_metrics: pd.DataFrame,
    rollrate_metrics: pd.DataFrame,
) -> pd.DataFrame:
    quantbullet = quantbullet_metrics[
        ["period", "cpr", "cdr", "cumulative_loss_rate"]
    ].rename(
        columns={
            "cpr": "quantbullet_cpr",
            "cdr": "quantbullet_cdr",
            "cumulative_loss_rate": "quantbullet_cgl",
        }
    )
    rollrate = rollrate_metrics[["period", "cpr", "cdr", "cgl"]].rename(
        columns={
            "cpr": "rollrate_cpr",
            "cdr": "rollrate_cdr",
            "cgl": "rollrate_cgl",
        }
    )
    diff = quantbullet.merge(rollrate, on="period", how="outer").sort_values("period")
    for field in ["cpr", "cdr", "cgl"]:
        diff[f"{field}_diff"] = diff[f"quantbullet_{field}"] - diff[f"rollrate_{field}"]
        diff[f"abs_{field}_diff"] = diff[f"{field}_diff"].abs()
    return diff


def build_summary(metric_diff: pd.DataFrame, cashflow_diff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in ["cpr", "cdr", "cgl"]:
        rows.append(
            {
                "metric": f"max_abs_{field}_diff",
                "value": float(metric_diff[f"abs_{field}_diff"].max()),
            }
        )
        rows.append(
            {
                "metric": f"mean_abs_{field}_diff",
                "value": float(metric_diff[f"abs_{field}_diff"].mean()),
            }
        )
    for field in ["begin_balance", "end_balance", "interest", "principal", "loss"]:
        rows.append(
            {
                "metric": f"diagnostic_max_abs_{field}_diff",
                "value": float(cashflow_diff[f"{field}_diff"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def write_excel(
    output_path: Path,
    *,
    summary: pd.DataFrame,
    cashflow_diff: pd.DataFrame,
    metric_diff: pd.DataFrame,
    quantbullet_cashflows: pd.DataFrame,
    rollrate_cashflows: pd.DataFrame,
    quantbullet_metrics: pd.DataFrame,
    rollrate_metrics: pd.DataFrame,
    run_config: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        run_config.to_excel(writer, sheet_name="run_config", index=False)
        metric_diff.to_excel(writer, sheet_name="metric_diff", index=False)
        cashflow_diff.to_excel(writer, sheet_name="cashflow_diff", index=False)
        quantbullet_metrics.to_excel(writer, sheet_name="quantbullet_metrics", index=False)
        rollrate_metrics.to_excel(writer, sheet_name="rollrate_metrics", index=False)
        quantbullet_cashflows.to_excel(writer, sheet_name="quantbullet_cashflows", index=False)
        rollrate_cashflows.to_excel(writer, sheet_name="rollrate_cashflows", index=False)


def _add_rollrate_python_path(roll_rate_root: Path) -> None:
    python_dir = roll_rate_root / "python"
    if not python_dir.exists():
        raise FileNotFoundError(f"roll-rate-model python directory not found: {python_dir}")
    sys.path.insert(0, str(python_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small PAR_2025_1 + GENERIC_v4 cashflow benchmark."
    )
    parser.add_argument("--config", type=Path, default=EXAMPLE_DIR / "config.json")
    parser.add_argument(
        "--roll-rate-root",
        type=Path,
        default=None,
        help="Path to roll-rate-model. Can also use ROLL_RATE_MODEL_ROOT.",
    )
    parser.add_argument("--max-loans", type=int, default=20)
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _resolve_roll_rate_root(roll_rate_root: Path | None) -> Path:
    if roll_rate_root is not None:
        return roll_rate_root

    env_value = os.environ.get("ROLL_RATE_MODEL_ROOT")
    if env_value:
        return Path(env_value)

    raise SystemExit(
        "roll-rate-model repo path is required. Provide --roll-rate-root "
        "or set ROLL_RATE_MODEL_ROOT."
    )


def main() -> None:
    args = parse_args()
    start = pd.Timestamp.now()
    config = load_config(args.config)
    roll_rate_root = _resolve_roll_rate_root(args.roll_rate_root)
    coefficient_root = roll_rate_root / "input" / "coef" / config["coefficient_set"]
    payment_matrix_path = roll_rate_root / "input" / "pmt_matrix.txt"
    prepped_loans = load_prepped_loans(
        roll_rate_root / config["loans_prepped_path"]
    )
    prepped_loans = prepped_loans[: args.max_loans]
    enriched_loans = [init_runtime_feature_state(loan, config) for loan in prepped_loans]
    quantbullet_loans = to_quantbullet_loans(enriched_loans)

    quantbullet_path_cashflows, quantbullet_cashflows = run_quantbullet_cashflows(
        quantbullet_loans,
        config=config,
        coefficient_root=coefficient_root,
        payment_matrix_path=payment_matrix_path,
        horizon=args.horizon,
        n_paths=args.n_paths,
        seed=args.seed,
    )
    rollrate_cashflows, rollrate_metrics = run_rollrate_cashflows(
        enriched_loans,
        roll_rate_root=roll_rate_root,
        config=config,
        coefficient_root=coefficient_root,
        payment_matrix_path=payment_matrix_path,
        horizon=args.horizon,
        n_paths=args.n_paths,
        seed=args.seed,
    )
    quantbullet_metrics = compute_period_metrics(quantbullet_cashflows)
    quantbullet_metrics = quantbullet_metrics.merge(
        rollrate_style_cpr_from_quantbullet_paths(
            quantbullet_path_cashflows,
            quantbullet_loans,
        ),
        on="period",
        how="left",
    )
    quantbullet_metrics["official_cpr"] = quantbullet_metrics["cpr"]
    quantbullet_metrics["cpr"] = quantbullet_metrics["rollrate_style_cpr"]
    cashflow_diff = compare_cashflows(quantbullet_cashflows, rollrate_cashflows)
    metric_diff = compare_metrics(quantbullet_metrics, rollrate_metrics)
    summary = build_summary(metric_diff, cashflow_diff)
    output_path = args.output or (EXAMPLE_DIR / "cashflow_benchmark.xlsx")
    run_config = pd.DataFrame(
        [
            {"key": "max_loans", "value": args.max_loans},
            {"key": "n_paths", "value": args.n_paths},
            {"key": "horizon", "value": args.horizon},
            {"key": "seed", "value": args.seed},
            {"key": "elapsed_seconds", "value": (pd.Timestamp.now() - start).total_seconds()},
        ]
    )
    write_excel(
        output_path,
        summary=summary,
        cashflow_diff=cashflow_diff,
        metric_diff=metric_diff,
        quantbullet_cashflows=quantbullet_cashflows,
        rollrate_cashflows=rollrate_cashflows,
        quantbullet_metrics=quantbullet_metrics,
        rollrate_metrics=rollrate_metrics,
        run_config=run_config,
    )
    print(summary.to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
