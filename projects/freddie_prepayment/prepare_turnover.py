"""Stage 1: select the cohort and derive model-specific target, weights and caps."""

import argparse
import polars as pl

from .common import CATEGORICAL, NUMERIC, TARGET, Config, read_config, temporary_output, write_summary


QUALITY_EXCLUSIONS = (
    "is_post_exit", "is_unknown_exit_code", "is_missing_exit_month",
    "is_event_month_mismatch", "is_zero_balance_without_exit",
)
REFERENCE = ("loan_identifier", "d_reporting_month", "d_origination_month", "vintage", "c_factor", "c_orig_ltv")


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
        *(pl.col(name).clip(spec.lower, spec.upper).cast(pl.Float32).alias(name + "_fit") for name, spec in NUMERIC.items()),
        *(pl.col(name).fill_null("MISSING") for name in CATEGORICAL),
        (pl.col("c_prev_balance") / pl.col("c_prev_balance").mean()).alias("weight"),
    ).with_row_index("row_id")


def prepare(config: Config) -> dict:
    source = pl.scan_parquet(config.panel_path)
    filters = cohort_filters(config.incentive_max)
    mask = pl.lit(True)
    expressions = [pl.len().alias("input")]
    for name, condition in filters:
        mask = mask & condition.fill_null(False)
        expressions.append(mask.sum().alias(name))
    counts = source.select(expressions).collect(engine="streaming").row(0, named=True)
    waterfall, previous = [], counts["input"]
    for name, remaining in counts.items():
        waterfall.append({"stage": name, "rows": remaining, "removed": previous - remaining})
        previous = remaining
    if previous == 0:
        raise ValueError("No complete observations in the turnover cohort")

    # Background bands use the same risk/quality/target rules, before the incentive cut.
    background = source
    for _, condition in filters[:5]:
        background = background.filter(condition)
    background = add_target(background).with_columns(
        pl.when(pl.col("c_incentive").is_null() | ~pl.col("c_incentive").is_finite()).then(pl.lit("missing"))
        .when(pl.col("c_incentive") <= -1).then(pl.lit("<= -1"))
        .when(pl.col("c_incentive") <= -.5).then(pl.lit("(-1, -0.5]"))
        .when(pl.col("c_incentive") <= 0).then(pl.lit("(-0.5, 0]"))
        .otherwise(pl.lit("> 0")).alias("band"),
    )
    bands = background.group_by("band").agg(
        pl.len().alias("rows"), pl.col(TARGET).sum().alias("events"),
        pl.col(TARGET).mean().alias("loan_month_rate"),
        ((pl.col(TARGET) * pl.col("c_prev_balance")).sum() / pl.col("c_prev_balance").sum()).alias("balance_weighted_rate"),
    ).collect(engine="streaming").sort("band").to_dicts()

    target = config.output_root / "turnover_frame.parquet"
    with temporary_output(target) as path:
        prepare_frame(source, incentive_max=config.incentive_max).sink_parquet(path, compression="zstd", row_group_size=250000)
        frame = pl.scan_parquet(path)
        stats = frame.select(
            pl.len().alias("rows"), pl.col("loan_identifier").n_unique().alias("loans"),
            pl.col(TARGET).sum().alias("events"), pl.col("weight").mean().alias("mean_weight"),
            pl.col("d_reporting_month").min().alias("first_month"), pl.col("d_reporting_month").max().alias("last_month"),
        ).collect(engine="streaming").row(0, named=True)
        if stats["rows"] != previous:
            raise ValueError("Prepared row count differs from the filter waterfall")
        clipping = frame.select(*[
            ((pl.col(name) < spec.lower) | (pl.col(name) > spec.upper)).sum().alias(name)
            for name, spec in NUMERIC.items()
        ]).collect(engine="streaming").row(0, named=True)
    summary = {"prepare": {**stats, "first_month": stats["first_month"].isoformat(),
                            "last_month": stats["last_month"].isoformat(),
                            "source": config.panel_path.name, "incentive_max": config.incentive_max,
                            "waterfall": waterfall, "clip_counts": clipping, "incentive_bands": bands}}
    write_summary(config.output_root, summary)
    print(f"[prepare] {stats['rows']:,} rows; {stats['loans']:,} loans; {stats['events']:,.0f} events", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    prepare(read_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
