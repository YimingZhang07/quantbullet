"""A/E of the turnover model by current balance x incentive band, overall and by reporting era.

The report's facet chart (7c, incentive curves by current balance) shows raw curves. This table shows whether
actual / expected tilts with incentive differently by balance, and how much of each cell comes from 2017-19.
Reads the prepared frame and the predictions under the config's output_root; prints tables only.

Run from the repository root (one line):
    .venv/Scripts/python.exe explorations/2026-10-05-balance-incentive-2d/ae_by_balance_incentive.py
        --config projects/freddie_prepayment/turnover.example.toml
"""
import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from projects.freddie_prepayment.config import read_config  # noqa: E402
from projects.freddie_prepayment.fit_turnover import CLIP  # noqa: E402

KEYS = ["row_id", "loan_identifier", "d_reporting_month"]
BREAKS = [100000, 150000, 250000, 350000, 500000, 700000]
BALANCE = ["<=100K", "100-150K", "150-250K", "250-350K", "350-500K", "500-700K", ">700K"]
INCENTIVE = ["[-5,-4)", "[-4,-3)", "[-3,-2)", "[-2,-1)", "[-1,-.5]"]


def load(root: Path) -> pl.DataFrame:
    return (pl.scan_parquet(root / "turnover_frame.parquet")
            .select(*KEYS, "c_prev_balance", "c_incentive", "y_full_prepay")
            .join(pl.scan_parquet(root / "turnover_predictions.parquet"), on=KEYS)
            .with_columns(
                pl.col("c_prev_balance").cut(BREAKS, labels=BALANCE).alias("balance"),
                pl.col("c_incentive").clip(*CLIP["c_incentive"])
                .cut([-4, -3, -2, -1], labels=INCENTIVE, left_closed=True).alias("incentive"),
                pl.when(pl.col("d_reporting_month").dt.year() <= 2019).then(pl.lit("2017-19"))
                .when(pl.col("d_reporting_month").dt.year() <= 2021).then(pl.lit("2020-21"))
                .otherwise(pl.lit("2022+")).alias("era"))
            .collect())


def cells(frame: pl.DataFrame) -> pl.DataFrame:
    """Balance-weighted A/E, events and the 2017-19 row share per balance x incentive cell."""
    w = pl.col("c_prev_balance")
    return frame.group_by("balance", "incentive").agg(
        ((w * pl.col("y_full_prepay")).sum() / (w * pl.col("pred_turnover")).sum()).round(2).alias("ae"),
        pl.col("y_full_prepay").sum().alias("events"),
        (pl.col("era") == "2017-19").mean().round(2).alias("share_2017_19"))


def pivot(table: pl.DataFrame, value: str) -> pl.DataFrame:
    wide = table.with_columns(pl.col("balance", "incentive").cast(pl.String)).pivot(
        on="incentive", index="balance", values=value)
    order = pl.DataFrame({"balance": BALANCE})
    return order.join(wide, on="balance", how="left").select(
        "balance", *[name for name in INCENTIVE if name in wide.columns])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    frame = load(read_config(args.config).output_root)
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_cols(10)
    print(f"{frame.height:,} rows")
    for label, subset in (("all", frame), ("2022+", frame.filter(pl.col("era") == "2022+")),
                          ("2017-19", frame.filter(pl.col("era") == "2017-19"))):
        table = cells(subset)
        print(f"\n== {label}: A/E (balance weighted)")
        print(pivot(table, "ae"))
        print(f"-- {label}: events")
        print(pivot(table, "events"))
    print("\n== share of 2017-19 rows per cell")
    print(pivot(cells(frame), "share_2017_19"))


if __name__ == "__main__":
    main()
