"""Compare QuantBullet and roll-rate production portfolio workbooks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_QUANTBULLET_WORKBOOK = EXAMPLE_DIR / "quantbullet_cashflows.xlsx"
DEFAULT_OUTPUT = EXAMPLE_DIR / "prod_comparison.xlsx"


def compare_ratios(
    quantbullet_workbook: Path,
    rollrate_workbook: Path,
) -> pd.DataFrame:
    quantbullet = pd.read_excel(
        quantbullet_workbook,
        sheet_name="portfolio_metrics",
    )[["period", "cpr", "cdr", "cumulative_loss_rate"]].rename(
        columns={
            "cpr": "quantbullet_cpr",
            "cdr": "quantbullet_cdr",
            "cumulative_loss_rate": "quantbullet_cgl",
        }
    )
    rollrate = pd.read_excel(
        rollrate_workbook,
        sheet_name="Metrics_Portfolio",
    )[["period", "cpr", "cdr", "cgl"]].rename(
        columns={
            "cpr": "rollrate_cpr",
            "cdr": "rollrate_cdr",
            "cgl": "rollrate_cgl",
        }
    )
    comparison = quantbullet.merge(rollrate, on="period", how="outer").sort_values(
        "period"
    )
    for metric in ("cpr", "cdr", "cgl"):
        comparison[f"{metric}_diff_bps"] = (
            comparison[f"quantbullet_{metric}"]
            - comparison[f"rollrate_{metric}"]
        ) * 10_000
    return comparison


def compare_cashflows(
    quantbullet_workbook: Path,
    rollrate_workbook: Path,
) -> pd.DataFrame:
    quantbullet = pd.read_excel(
        quantbullet_workbook,
        sheet_name="portfolio_cashflows",
    )[
        [
            "period",
            "begin_balance",
            "end_balance",
            "interest_collected",
            "principal_collected",
            "loss",
        ]
    ]
    rollrate = pd.read_excel(
        rollrate_workbook,
        sheet_name="Portfolio",
    )[
        ["period", "begin_bal", "end_bal", "int_pmt", "prin_pmt", "loss"]
    ].rename(
        columns={
            "begin_bal": "begin_balance",
            "end_bal": "end_balance",
            "int_pmt": "interest_collected",
            "prin_pmt": "principal_collected",
        }
    )
    comparison = quantbullet.merge(
        rollrate,
        on="period",
        suffixes=("_quantbullet", "_rollrate"),
    )
    initial_balance = float(quantbullet["begin_balance"].iloc[0])
    for field in (
        "begin_balance",
        "end_balance",
        "interest_collected",
        "principal_collected",
        "loss",
    ):
        difference = (
            comparison[f"{field}_quantbullet"]
            - comparison[f"{field}_rollrate"]
        )
        comparison[f"{field}_diff"] = difference
        comparison[f"{field}_diff_bps_initial"] = (
            difference / initial_balance * 10_000
        )
    return comparison


def summarize(
    ratio_diff: pd.DataFrame,
    cashflow_diff: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for metric in ("cpr", "cdr", "cgl"):
        difference = ratio_diff[f"{metric}_diff_bps"]
        rows.extend(
            [
                {
                    "metric": f"mean_abs_{metric}_diff_bps",
                    "value": float(difference.abs().mean()),
                },
                {
                    "metric": f"max_abs_{metric}_diff_bps",
                    "value": float(difference.abs().max()),
                },
                {
                    "metric": f"ending_{metric}_diff_bps",
                    "value": float(difference.iloc[-1]),
                },
            ]
        )
    for field in (
        "begin_balance",
        "end_balance",
        "interest_collected",
        "principal_collected",
        "loss",
    ):
        rows.append(
            {
                "metric": f"max_abs_{field}_diff_bps_initial",
                "value": float(
                    cashflow_diff[f"{field}_diff_bps_initial"].abs().max()
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare QuantBullet and roll-rate production workbooks."
    )
    parser.add_argument(
        "--quantbullet-workbook",
        type=Path,
        default=DEFAULT_QUANTBULLET_WORKBOOK,
    )
    parser.add_argument("--rollrate-workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratio_diff = compare_ratios(
        args.quantbullet_workbook,
        args.rollrate_workbook,
    )
    cashflow_diff = compare_cashflows(
        args.quantbullet_workbook,
        args.rollrate_workbook,
    )
    summary = summarize(ratio_diff, cashflow_diff)
    sources = pd.DataFrame(
        [
            {
                "framework": "QuantBullet",
                "workbook": str(args.quantbullet_workbook),
            },
            {
                "framework": "roll-rate prod",
                "workbook": str(args.rollrate_workbook),
            },
        ]
    )
    with pd.ExcelWriter(args.output) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        ratio_diff.to_excel(writer, sheet_name="ratio_diff", index=False)
        cashflow_diff.to_excel(writer, sheet_name="cashflow_diff", index=False)
        sources.to_excel(writer, sheet_name="sources", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
