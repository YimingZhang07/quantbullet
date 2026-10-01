"""Read original ZHVI wide CSVs as monthly long tables."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import csv
import re

import polars as pl

from . import _validate_zhvi_header


_METADATA = {
    "RegionID": "region_id", "RegionName": "region_name", "RegionType": "region_type",
    "SizeRank": "size_rank", "StateName": "state_name", "State": "state",
    "City": "city", "Metro": "metro", "CountyName": "county_name",
}


def scan_zhvi_csv(path: str | Path, *, geography: str) -> pl.LazyFrame:
    """Preserve every source region/month cell, including null ZHVI values.

    ``geography`` identifies the source file (metro, state, or zip). The
    national row in the metro file is labelled national. IDs and ZIP names
    remain strings; source StateName and State are preserved without mapping.
    Dates become month starts, and dollar values are parsed as Float64.
    """
    mappings = {
        "metro": {"country": "national", "msa": "metro"},
        "state": {"state": "state"}, "zip": {"zip": "zip"},
    }
    if geography not in mappings:
        raise ValueError("ZHVI source geography must be metro, state, or zip")
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        header = tuple(next(csv.reader(file)))
    if len(set(header)) != len(header):
        raise ValueError("Duplicate ZHVI source columns")
    _validate_zhvi_header(header)
    months = [name for name in header if re.fullmatch(r"\d{4}-\d{2}-\d{2}", name)]
    normalized = [date.fromisoformat(name).replace(day=1) for name in months]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Multiple ZHVI columns normalize to the same month")
    frame = pl.scan_csv(path, infer_schema=False, null_values=[""])
    metadata = [
        (pl.col(source) if source in header else pl.lit(None, dtype=pl.String)).alias(target)
        for source, target in _METADATA.items()
    ]
    frame = frame.select(*metadata, *[pl.col(month) for month in months]).unpivot(
        on=months, index=list(_METADATA.values()), variable_name="source_month", value_name="value",
    )
    return frame.with_columns(
        pl.lit("zillow").alias("provider"),
        pl.lit("ZHVI").alias("metric"),
        pl.col("region_type").replace_strict(mappings[geography], return_dtype=pl.String).alias("geography_level"),
        pl.col("source_month").str.to_date("%Y-%m-%d", strict=True).dt.month_start().alias("month"),
        pl.col("value").cast(pl.Float64, strict=True),
        pl.col("size_rank").cast(pl.Int64, strict=True),
    ).select(
        "provider", "metric", "geography_level", "region_id", "region_name", "month", "value",
        "region_type", "size_rank", "state_name", "state", "city", "metro", "county_name",
    )
