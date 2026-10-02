"""Build fixed HPI and CPI Parquet tables from the current downloaded CSVs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import polars as pl

from quantbullet.data.fred import scan_cpi_csv
from quantbullet.data.zillow import scan_zhvi_csv


def _source_paths(root: Path) -> dict[str, Path]:
    datasets = json.loads((root / "manifests/downloads.json").read_text(encoding="utf-8"))["datasets"]
    sources = {}
    for name in ("zhvi_metro", "zhvi_state", "zhvi_zip", "cpi"):
        relative = Path(datasets[name]["current"]["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"{name}: source path must be relative and inside the data root")
        sources[name] = path
    return sources


def _validate(frame: pl.LazyFrame, *, hpi: bool) -> dict:
    keys = ["provider", "metric", "geography_level", "region_id"] if hpi else ["series_id"]
    missing_id = pl.any_horizontal(*[
        pl.col(key).is_null() | (pl.col(key).str.strip_chars() == "") for key in keys
    ])
    summary = frame.select(
        pl.len().alias("rows"),
        pl.col("month").min().alias("first_month"),
        pl.col("month").max().alias("last_month"),
        missing_id.sum().alias("missing_id"),
        pl.col("month").is_null().sum().alias("missing_month"),
        (pl.col("month").dt.day() != 1).sum().alias("wrong_day"),
        (pl.col("value").is_not_null() & ~pl.col("value").is_finite()).sum().alias("nonfinite"),
    ).collect(engine="streaming").row(0, named=True)
    errors = ("missing_id", "missing_month", "wrong_day", "nonfinite")
    if not summary["rows"] or any(summary[key] for key in errors):
        raise ValueError(f"{'HPI' if hpi else 'CPI'} validation failed: {summary}")
    duplicates = frame.group_by(keys).agg(
        pl.len().alias("rows"), pl.col("month").n_unique().alias("months"),
    ).filter(
        pl.col("rows") != pl.col("months")
    ).limit(5).collect(engine="streaming")
    if duplicates.height:
        raise ValueError(f"{'HPI' if hpi else 'CPI'} duplicate keys")
    if hpi:
        invalid_zip = frame.filter((pl.col("geography_level") == "zip") & (
            pl.col("region_name").is_null() | ~pl.col("region_name").str.contains(r"^\d{5}$")
        )).limit(1).collect(engine="streaming")
        if invalid_zip.height:
            raise ValueError("ZIP RegionName must be a five-digit string")
    return {key: summary[key] for key in ("rows", "first_month", "last_month")}


@contextmanager
def _staging_directory(scratch: Path):
    temporary = tempfile.TemporaryDirectory(prefix="normalize-", dir=scratch)
    try:
        yield Path(temporary.name)
    finally:
        # A failed streaming sink can briefly retain a Windows file handle.
        for attempt in range(5):
            try:
                temporary.cleanup()
                break
            except OSError as exc:
                transient = (
                    isinstance(exc, PermissionError)
                    or exc.errno == errno.ENOTEMPTY
                    or getattr(exc, "winerror", None) in {32, 145}
                )
                if attempt == 4 or not transient:
                    raise
                time.sleep(0.1 * (attempt + 1))


def build_parquet(data_root: str | Path) -> dict:
    """Rebuild both tables; validate and finish both temporary files before replacing outputs."""
    root = Path(data_root).expanduser().resolve()
    sources = _source_paths(root)
    hpi = pl.concat([
        scan_zhvi_csv(sources[f"zhvi_{level}"], geography=level)
        for level in ("metro", "state", "zip")
    ])
    cpi = scan_cpi_csv(sources["cpi"])
    summaries = {"hpi": _validate(hpi, hpi=True), "cpi": _validate(cpi, hpi=False)}
    output = root / "parquet"
    output.mkdir(parents=True, exist_ok=True)
    with _staging_directory(output) as temporary:
        hpi.sink_parquet(temporary / "hpi.parquet", compression="zstd", row_group_size=250_000)
        cpi.sink_parquet(temporary / "cpi.parquet", compression="zstd")
        for name in ("hpi", "cpi"):
            (temporary / f"{name}.parquet").replace(output / f"{name}.parquet")
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("MACRO_DATA_ROOT"))
    args = parser.parse_args(argv)
    if args.data_root is None:
        parser.error("Set MACRO_DATA_ROOT or pass --data-root")
    root = args.data_root.expanduser().resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Data root must be outside the repository")
    try:
        summaries = build_parquet(root)
    except Exception as exc:
        print(f"[housing_macro] FAILED: {exc}", file=sys.stderr)
        return 1
    for name, summary in summaries.items():
        print(f"[{name}] {summary['rows']:,} rows, {summary['first_month']} to {summary['last_month']}, parquet/{name}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
