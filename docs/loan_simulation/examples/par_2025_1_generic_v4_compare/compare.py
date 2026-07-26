"""Compare QuantBullet and roll-rate generated workbooks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXAMPLE_DIR = Path(__file__).resolve().parent
QUANTBULLET_WORKBOOK = (
    EXAMPLE_DIR.parent
    / "par_2025_1_generic_v4_quantbullet"
    / "quantbullet_cashflows.xlsx"
)
ROLLRATE_WORKBOOK = (
    EXAMPLE_DIR.parent
    / "par_2025_1_generic_v4_rollrate"
    / "rollrate_cashflows.xlsx"
)
OUTPUT_PATH = EXAMPLE_DIR / "comparison.xlsx"

ASSUMPTIONS = [
    {
        "assumption": "input_starting_point",
        "value": "Starts from roll-rate loans_prepped data, not raw deal tape.",
    },
    {
        "assumption": "coefficient_set",
        "value": "Uses GENERIC_v4 coefficient TSV files.",
    },
    {
        "assumption": "macro_overrides",
        "value": "Enabled for GENERIC_v4 runtime features via independently implemented CPI and FICO-coupon projection semantics.",
    },
    {
        "assumption": "dials_overlays",
        "value": "Disabled.",
    },
    {
        "assumption": "runtime_features",
        "value": "Age/date/month/days_to_month_end/month_group and macro model features are rolled independently on each side.",
    },
    {
        "assumption": "cpr",
        "value": "Comparison uses roll-rate-style CPR reconstructed from QuantBullet PIF event balances; official QuantBullet CPR remains in quantbullet_metrics.",
    },
]


def load_workbooks(
    quantbullet_workbook: Path,
    rollrate_workbook: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quantbullet_book = pd.ExcelFile(quantbullet_workbook)
    quantbullet_cashflows = pd.read_excel(quantbullet_book, sheet_name="portfolio_cashflows")
    quantbullet_metrics = pd.read_excel(quantbullet_book, sheet_name="portfolio_metrics")
    quantbullet_path_cashflows = pd.read_excel(quantbullet_book, sheet_name="path_cashflows")
    rollrate_cashflows = pd.read_excel(rollrate_workbook, sheet_name="cashflows")
    rollrate_metrics = pd.read_excel(rollrate_workbook, sheet_name="metrics")
    return (
        quantbullet_cashflows,
        quantbullet_metrics,
        quantbullet_path_cashflows,
        rollrate_cashflows,
        rollrate_metrics,
    )


def compare_cashflows(
    quantbullet_cashflows: pd.DataFrame,
    rollrate_cashflows: pd.DataFrame,
) -> pd.DataFrame:
    quantbullet = quantbullet_cashflows[
        [
            "period",
            "begin_balance",
            "end_balance",
            "interest_collected",
            "principal_collected",
            "loss",
        ]
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
        diff[f"abs_{field}_diff"] = diff[f"{field}_diff"].abs()
    return diff


def compare_status_counts(
    quantbullet_path_cashflows: pd.DataFrame,
    rollrate_cashflows: pd.DataFrame,
) -> pd.DataFrame:
    n_paths = quantbullet_path_cashflows["path_id"].nunique()
    qb_counts = (
        quantbullet_path_cashflows.groupby(["period", "end_status"], as_index=False)
        .size()
        .pivot(index="period", columns="end_status", values="size")
        .fillna(0.0)
        .div(n_paths)
        .reset_index()
    )
    qb_counts = qb_counts.rename(
        columns={status: f"quantbullet_count_{status}" for status in qb_counts.columns if status != "period"}
    )

    rr = rollrate_cashflows.copy()
    rr_counts = pd.DataFrame({"period": rr["period"]})
    rr_counts["rollrate_count_PIF"] = rr.get("pif_cnt", 0.0)
    rr_counts["rollrate_count_LIQ"] = rr.get("liq_cnt", 0.0)
    rr_counts["rollrate_count_D1M"] = rr.get("dq30", 0.0)
    rr_counts["rollrate_count_D2M"] = rr.get("dq60", 0.0)
    rr_counts["rollrate_count_D3M"] = rr.get("dq90", 0.0)
    rr_counts["rollrate_count_D4M"] = rr.get("dq120", 0.0)
    non_current = [
        "rollrate_count_PIF",
        "rollrate_count_LIQ",
        "rollrate_count_D1M",
        "rollrate_count_D2M",
        "rollrate_count_D3M",
        "rollrate_count_D4M",
    ]
    rr_counts["rollrate_count_C"] = rr["cnt"] - rr_counts[non_current].sum(axis=1)

    diff = qb_counts.merge(rr_counts, on="period", how="outer").fillna(0.0)
    for status in ["C", "D1M", "D2M", "D3M", "D4M", "PIF", "LIQ"]:
        q_col = f"quantbullet_count_{status}"
        r_col = f"rollrate_count_{status}"
        if q_col not in diff:
            diff[q_col] = 0.0
        if r_col not in diff:
            diff[r_col] = 0.0
        diff[f"{status}_count_diff"] = diff[q_col] - diff[r_col]
        diff[f"abs_{status}_count_diff"] = diff[f"{status}_count_diff"].abs()
    return diff.sort_values("period").reset_index(drop=True)


def rollrate_style_cpr_from_quantbullet_paths(
    quantbullet_path_cashflows: pd.DataFrame,
    rollrate_cashflows: pd.DataFrame,
) -> pd.DataFrame:
    frame = quantbullet_path_cashflows.copy()
    frame["payoff_event_balance"] = frame["begin_balance"].where(
        frame["end_status"].eq("PIF"),
        0.0,
    )
    grouped = (
        frame.groupby("period", as_index=False)["payoff_event_balance"]
        .sum()
        .sort_values("period")
        .reset_index(drop=True)
    )
    n_paths = quantbullet_path_cashflows["path_id"].nunique()
    grouped["payoff_event_balance"] = grouped["payoff_event_balance"] / n_paths
    rollrate_denominator = rollrate_cashflows[["period", "begin_bal", "sch_prin"]]
    grouped = grouped.merge(rollrate_denominator, on="period", how="left")
    denominator = (grouped["begin_bal"] - grouped["sch_prin"]).clip(lower=0.0)
    smm = (grouped["payoff_event_balance"] / denominator.replace(0, float("nan"))).fillna(
        0.0
    )
    grouped["rollrate_style_cpr"] = 1.0 - (1.0 - smm.clip(upper=1.0)) ** 12
    return grouped[["period", "rollrate_style_cpr"]]


def compare_metrics(
    quantbullet_metrics: pd.DataFrame,
    quantbullet_path_cashflows: pd.DataFrame,
    rollrate_metrics: pd.DataFrame,
    rollrate_cashflows: pd.DataFrame,
) -> pd.DataFrame:
    quantbullet = quantbullet_metrics[
        ["period", "cpr", "cdr", "cumulative_loss_rate"]
    ].rename(
        columns={
            "cpr": "quantbullet_official_cpr",
            "cdr": "quantbullet_cdr",
            "cumulative_loss_rate": "quantbullet_cgl",
        }
    )
    quantbullet = quantbullet.merge(
        rollrate_style_cpr_from_quantbullet_paths(
            quantbullet_path_cashflows,
            rollrate_cashflows,
        ),
        on="period",
        how="left",
    )
    quantbullet["quantbullet_cpr"] = quantbullet["rollrate_style_cpr"]
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


def build_summary(
    metric_diff: pd.DataFrame,
    cashflow_diff: pd.DataFrame,
    status_count_diff: pd.DataFrame,
) -> pd.DataFrame:
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
                "metric": f"max_abs_{field}_diff",
                "value": float(cashflow_diff[f"abs_{field}_diff"].max()),
            }
        )
    rows.append(
        {
            "metric": "max_abs_status_count_diff",
            "value": float(
                status_count_diff[
                    [
                        column
                        for column in status_count_diff.columns
                        if column.startswith("abs_") and column.endswith("_count_diff")
                    ]
                ]
                .max()
                .max()
            ),
        }
    )
    rows.append(
        {
            "metric": "cpr_note",
            "value": "CPR diff uses QuantBullet PIF event balance from path_cashflows with roll-rate scheduled-principal denominator; official CPR is preserved as quantbullet_official_cpr.",
        }
    )
    return pd.DataFrame(rows)


def write_excel(
    output_path: Path,
    *,
    summary: pd.DataFrame,
    metric_diff: pd.DataFrame,
    cashflow_diff: pd.DataFrame,
    status_count_diff: pd.DataFrame,
    quantbullet_metrics: pd.DataFrame,
    rollrate_metrics: pd.DataFrame,
    quantbullet_cashflows: pd.DataFrame,
    rollrate_cashflows: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(ASSUMPTIONS).to_excel(
            writer,
            sheet_name="assumptions",
            index=False,
        )
        metric_diff.to_excel(writer, sheet_name="metric_diff", index=False)
        status_count_diff.to_excel(writer, sheet_name="status_count_diff", index=False)
        cashflow_diff.to_excel(writer, sheet_name="cashflow_diff", index=False)
        quantbullet_metrics.to_excel(writer, sheet_name="quantbullet_metrics", index=False)
        rollrate_metrics.to_excel(writer, sheet_name="rollrate_metrics", index=False)
        quantbullet_cashflows.to_excel(writer, sheet_name="quantbullet_cashflows", index=False)
        rollrate_cashflows.to_excel(writer, sheet_name="rollrate_cashflows", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated example workbooks.")
    parser.add_argument("--quantbullet-workbook", type=Path, default=QUANTBULLET_WORKBOOK)
    parser.add_argument("--rollrate-workbook", type=Path, default=ROLLRATE_WORKBOOK)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (
        quantbullet_cashflows,
        quantbullet_metrics,
        quantbullet_path_cashflows,
        rollrate_cashflows,
        rollrate_metrics,
    ) = load_workbooks(args.quantbullet_workbook, args.rollrate_workbook)
    metric_diff = compare_metrics(
        quantbullet_metrics,
        quantbullet_path_cashflows,
        rollrate_metrics,
        rollrate_cashflows,
    )
    cashflow_diff = compare_cashflows(quantbullet_cashflows, rollrate_cashflows)
    status_count_diff = compare_status_counts(
        quantbullet_path_cashflows,
        rollrate_cashflows,
    )
    summary = build_summary(metric_diff, cashflow_diff, status_count_diff)
    write_excel(
        args.output,
        summary=summary,
        metric_diff=metric_diff,
        cashflow_diff=cashflow_diff,
        status_count_diff=status_count_diff,
        quantbullet_metrics=quantbullet_metrics,
        rollrate_metrics=rollrate_metrics,
        quantbullet_cashflows=quantbullet_cashflows,
        rollrate_cashflows=rollrate_cashflows,
    )
    print(summary.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
