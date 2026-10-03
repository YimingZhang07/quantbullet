"""Read CPI and weekly PMMS CSVs distributed by FRED."""

from __future__ import annotations

from pathlib import Path
from datetime import date
import csv

import polars as pl

from . import _validate_fred_header, fred_csv_source


def _scan_fred_values(path: str | Path, series_id: str) -> pl.LazyFrame:
    fred_csv_source(series_id)  # Validate the identifier before referencing a column.
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        header = tuple(next(csv.reader(file)))
    if len(set(header)) != len(header):
        raise ValueError("Duplicate FRED source columns")
    _validate_fred_header(header, series_id=series_id)
    return pl.scan_csv(path, infer_schema=False, null_values=["", "."]).select(
        pl.col("observation_date").str.to_date("%Y-%m-%d", strict=True),
        pl.col(series_id).cast(pl.Float64, strict=True).alias("value"),
    )


def scan_cpi_csv(path: str | Path, *, series_id: str = "CPIAUCNS") -> pl.LazyFrame:
    """Keep each CPI observation, including nulls; normalize dates to month starts."""
    return _scan_fred_values(path, series_id).select(
        pl.lit("bls").alias("provider"),
        pl.lit(series_id).alias("series_id"),
        pl.col("observation_date").dt.month_start().alias("month"), "value",
    )


def scan_pmms_csv(path: str | Path, *, series_id: str = "MORTGAGE30US") -> pl.LazyFrame:
    """Read 30-year PMMS weekly dates and rates in percent, without date shifting."""
    if series_id != "MORTGAGE30US":
        raise ValueError("PMMS reader currently supports MORTGAGE30US only")
    return _scan_fred_values(path, series_id).select(
        pl.lit("freddie_mac").alias("provider"), pl.lit(series_id).alias("series_id"),
        "observation_date", "value",
    )


def aggregate_pmms_monthly(weekly: pl.LazyFrame, *, as_of: date) -> pl.LazyFrame:
    """Validate all weekly records, then average valid values in elapsed months.

    The caller supplies the cutoff date. Its calendar month is excluded;
    all-null months stay null and missing observations are not filled.
    Validation scans the small weekly table before returning a LazyFrame.
    """
    if type(as_of) is not date:
        raise ValueError("as_of must be a date")
    summary = weekly.select(
        pl.len().alias("rows"),
        (pl.col("series_id").is_null() | (pl.col("series_id").str.strip_chars() == "")).sum().alias("missing_id"),
        pl.col("observation_date").is_null().sum().alias("missing_date"),
        (pl.col("value").is_not_null() & ~pl.col("value").is_finite()).sum().alias("nonfinite"),
        pl.struct("series_id", "observation_date").n_unique().alias("unique_dates"),
    ).collect().row(0, named=True)
    if not summary["rows"] or any(summary[k] for k in ("missing_id", "missing_date", "nonfinite")):
        raise ValueError(f"PMMS weekly validation failed: {summary}")
    if summary["rows"] != summary["unique_dates"]:
        raise ValueError("PMMS duplicate weekly dates")
    return (
        weekly.with_columns(pl.col("observation_date").dt.month_start().alias("month"))
        .filter(pl.col("month") < as_of.replace(day=1))
        .group_by("provider", "series_id", "month").agg(pl.col("value").mean())
        .sort("series_id", "month")
    )
