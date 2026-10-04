"""Stage 1: filter the cohort, label categories, and derive the target."""

import argparse
import polars as pl

from quantbullet.utils.files import temporary_output

from .config import Config, read_config


QUALITY_EXCLUSIONS = (
    "is_post_exit", "is_unknown_exit_code", "is_missing_exit_month",
    "is_event_month_mismatch", "is_zero_balance_without_exit",
)
REFERENCE = ("loan_identifier", "d_reporting_month", "d_origination_month", "vintage", "c_factor", "c_orig_ltv")
TARGET = "y_full_prepay"
NUMERIC = ("c_age", "c_incentive", "c_orig_fico", "c_updated_ltv", "c_orig_balance", "c_hpi_growth")
CATEGORICAL = ("f_purpose", "f_occupancy", "f_property_type", "f_first_time_buyer", "f_month", "f_state")


def cohort_filters(incentive_max: float):
    # CURRENT is a beginning-of-month risk set. Do not select on the outcome status.
    return [
        ("previous_current", (pl.col("f_pre_status") == "CURRENT") & pl.col("is_consecutive_month")),
        ("positive_balance_and_age", (pl.col("c_prev_balance") > 0) & pl.col("c_prev_balance").is_finite() & (pl.col("c_age") >= 1)),
        ("unmodified", ~pl.col("is_ever_modified")),
        ("known_observation", ~pl.any_horizontal(*(pl.col(name) for name in QUALITY_EXCLUSIONS))),
        ("known_payoff_maturity", ~((pl.col("zero_balance_code") == "01") & pl.col("d_maturity_month").is_null()).fill_null(False)),
        ("turnover_incentive", pl.col("c_incentive").is_finite() & (pl.col("c_incentive") <= incentive_max)),
        ("complete_numeric_features", pl.all_horizontal(*(pl.col(name).is_not_null() & pl.col(name).is_finite() for name in NUMERIC))),
    ]


def add_target(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame.with_columns(
        ((pl.col("zero_balance_code") == "01")
         & (pl.col("d_exit_month") == pl.col("d_reporting_month"))
         & (pl.col("d_reporting_month") < pl.col("d_maturity_month")))
        .fill_null(False).cast(pl.Float64).alias(TARGET),
    )


def prepare_frame(panel: pl.LazyFrame, *, incentive_max: float = -.5) -> pl.LazyFrame:
    for _, condition in cohort_filters(incentive_max):
        panel = panel.filter(condition)
    return add_target(panel).select(
        *REFERENCE, "c_prev_balance", TARGET, *NUMERIC, *CATEGORICAL,
    ).with_columns(
        *(pl.col(name).fill_null("MISSING") for name in CATEGORICAL),
        (pl.col("c_prev_balance") / pl.col("c_prev_balance").mean()).alias("weight"),
    ).with_row_index("row_id")


def prepare(config: Config) -> dict:
    source = pl.scan_parquet(config.panel_path)
    mask = pl.lit(True)
    for _, condition in cohort_filters(config.incentive_max):
        mask = mask & condition.fill_null(False)
    expected = source.select(mask.sum().alias("rows")).collect(engine="streaming").item()
    if expected == 0:
        raise ValueError("No complete observations in the turnover cohort")

    target = config.output_root / "turnover_frame.parquet"
    with temporary_output(target) as path:
        prepare_frame(source, incentive_max=config.incentive_max).sink_parquet(path, compression="zstd", row_group_size=250000)
        stats = pl.scan_parquet(path).select(
            pl.len().alias("rows"), pl.col("loan_identifier").n_unique().alias("loans"),
            pl.col(TARGET).sum().alias("events"),
        ).collect(engine="streaming").row(0, named=True)
        if stats["rows"] != expected:
            raise ValueError("Prepared row count differs from the cohort filters")
    print(f"[prepare] {stats['rows']:,} rows; {stats['loans']:,} loans; {stats['events']:,.0f} events", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    prepare(read_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
