from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def compute_period_metrics(
    cashflows: pd.DataFrame,
    *,
    group_by: Sequence[str] | None = None,
    annualization_periods: int = 12,
) -> pd.DataFrame:
    """Compute period-level credit metrics from cashflow output.

    ``cashflows`` can be path-level, loan-level, or portfolio-level output from
    the simulator. Metrics are computed from summed balances and cashflow
    amounts within each period and optional grouping columns. ``original_balance``
    is required for cumulative loss denominators.
    """
    if cashflows.empty:
        return pd.DataFrame()
    if annualization_periods <= 0:
        raise ValueError("annualization_periods must be positive")

    group_columns = _metric_group_columns(cashflows, group_by=group_by)
    aggregated = _aggregate_cashflows(cashflows, group_columns=group_columns)
    period_group_columns = [col for col in group_columns if col not in {"period", "period_date"}]

    prepay_denominator = (
        aggregated["begin_balance"] - aggregated["scheduled_principal"]
    ).clip(lower=0.0)
    aggregated["smm"] = _safe_divide(
        aggregated["prepayment_amount"],
        prepay_denominator,
    ).clip(lower=0.0, upper=1.0)
    aggregated["cpr"] = 1.0 - (1.0 - aggregated["smm"]) ** annualization_periods

    aggregated["mdr"] = _safe_divide(
        aggregated["default_balance"],
        aggregated["begin_balance"],
    ).clip(lower=0.0, upper=1.0)
    aggregated["cdr"] = 1.0 - (1.0 - aggregated["mdr"]) ** annualization_periods

    aggregated["period_loss_rate"] = _safe_divide(
        aggregated["loss"],
        aggregated["begin_balance"],
    )
    aggregated["period_net_loss"] = aggregated["loss"] - aggregated["net_recovery"]
    aggregated["delinquency_rate"] = _safe_divide(
        aggregated["delinquent_balance"],
        aggregated["end_balance"],
    )

    if period_group_columns:
        group_key = period_group_columns
        aggregated["cumulative_loss"] = aggregated.groupby(group_key)["loss"].cumsum()
        aggregated["cumulative_net_loss"] = aggregated.groupby(group_key)[
            "period_net_loss"
        ].cumsum()
        original_balance = aggregated.groupby(group_key)["original_balance"].transform("first")
    else:
        aggregated["cumulative_loss"] = aggregated["loss"].cumsum()
        aggregated["cumulative_net_loss"] = aggregated["period_net_loss"].cumsum()
        original_balance = pd.Series(
            aggregated["original_balance"].iloc[0],
            index=aggregated.index,
        )

    aggregated["cumulative_loss_rate"] = _safe_divide(
        aggregated["cumulative_loss"],
        original_balance,
    )
    aggregated["cumulative_net_loss_rate"] = _safe_divide(
        aggregated["cumulative_net_loss"],
        original_balance,
    )

    return aggregated


def _metric_group_columns(
    cashflows: pd.DataFrame,
    *,
    group_by: Sequence[str] | None,
) -> list[str]:
    columns = list(group_by or [])
    if "period" not in columns:
        columns.append("period")
    if "period_date" in cashflows.columns and "period_date" not in columns:
        columns.append("period_date")
    missing = [column for column in columns if column not in cashflows.columns]
    if missing:
        raise KeyError(f"cashflows missing group columns: {missing}")
    return columns


def _aggregate_cashflows(
    cashflows: pd.DataFrame,
    *,
    group_columns: list[str],
) -> pd.DataFrame:
    required_columns = [
        "begin_balance",
        "end_balance",
        "scheduled_principal",
        "principal_collected",
        "prepayment_amount",
        "original_balance",
        "default_balance",
        "loss",
        "net_recovery",
        "delinquent_balance",
    ]
    missing = [column for column in required_columns if column not in cashflows.columns]
    if missing:
        raise KeyError(f"cashflows missing metric columns: {missing}")

    aggregated = (
        cashflows.groupby(group_columns, as_index=False)[required_columns]
        .sum()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return aggregated


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, pd.NA)
    return result.fillna(0.0)
