"""Read monthly CPI CSVs distributed by FRED."""

from __future__ import annotations

from pathlib import Path
import csv

import polars as pl

from . import _validate_fred_header, fred_csv_source


def scan_cpi_csv(path: str | Path, *, series_id: str = "CPIAUCNS") -> pl.LazyFrame:
    """Keep each observation, including blank or dot-valued months.

    The provider is BLS; FRED is the distributor recorded in the download
    manifest. Dates are normalized to month starts, not publication dates.
    """
    fred_csv_source(series_id)  # Validate the identifier before referencing a column.
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        header = tuple(next(csv.reader(file)))
    if len(set(header)) != len(header):
        raise ValueError("Duplicate CPI source columns")
    _validate_fred_header(header, series_id=series_id)
    return pl.scan_csv(path, infer_schema=False, null_values=["", "."]).select(
        pl.lit("bls").alias("provider"),
        pl.lit(series_id).alias("series_id"),
        pl.col("observation_date").str.to_date("%Y-%m-%d", strict=True).dt.month_start().alias("month"),
        pl.col(series_id).cast(pl.Float64, strict=True).alias("value"),
    )
