"""Build selected HPI, CPI, and monthly PMMS tables from current CSV snapshots."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import polars as pl

from quantbullet.data.fred import aggregate_pmms_monthly, scan_cpi_csv, scan_pmms_csv
from quantbullet.data.zillow import scan_zhvi_csv


SOURCE_DATASETS = {
    "hpi": ("zhvi_metro", "zhvi_state", "zhvi_zip"),
    "cpi": ("cpi",),
    "pmms": ("pmms_30y",),
}


def _source_paths(root: Path, datasets: dict, names: list[str]) -> dict[str, Path]:
    sources = {}
    for name in names:
        if name not in datasets:
            raise ValueError(f"{name}: missing current snapshot; run the download command first")
        relative = Path(datasets[name]["current"]["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"{name}: source path must be relative and inside the data root")
        if not path.is_file():
            raise ValueError(f"{name}: source file is missing")
        sources[name] = path
    return sources


def _validate(frame: pl.LazyFrame, *, table: str) -> dict:
    hpi = table == "hpi"
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
        raise ValueError(f"{table.upper()} validation failed: {summary}")
    duplicates = frame.group_by(keys).agg(
        pl.len().alias("rows"), pl.col("month").n_unique().alias("months"),
    ).filter(
        pl.col("rows") != pl.col("months")
    ).limit(5).collect(engine="streaming")
    if duplicates.height:
        raise ValueError(f"{table.upper()} duplicate keys")
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


def build_parquet(data_root: str | Path, *, datasets: Sequence[str] | None = None) -> dict:
    """Finish every selected table before replacing its fixed output file.

    PMMS excludes the month of snapshot acquisition and all later months,
    also bounded by the UTC build date. Cached data cannot age into a full month.
    """
    root = Path(data_root).expanduser().resolve()
    selected = list(SOURCE_DATASETS) if datasets is None else list(dict.fromkeys(datasets))
    if not selected or any(name not in SOURCE_DATASETS for name in selected):
        raise ValueError("Select one or more of hpi, cpi, pmms")
    entries = json.loads((root / "manifests/downloads.json").read_text(encoding="utf-8"))["datasets"]
    required = [source for name in selected for source in SOURCE_DATASETS[name]]
    sources = _source_paths(root, entries, required)
    frames = {}
    for name in selected:
        if name == "hpi":
            frames[name] = pl.concat([
                scan_zhvi_csv(sources[f"zhvi_{level}"], geography=level)
                for level in ("metro", "state", "zip")
            ])
        elif name == "cpi":
            frames[name] = scan_cpi_csv(sources["cpi"])
        else:
            acquired = datetime.fromisoformat(entries["pmms_30y"]["current"]["downloaded_at_utc"])
            if acquired.tzinfo is None:
                raise ValueError("pmms_30y: downloaded_at_utc must include a timezone")
            as_of = min(datetime.now(timezone.utc).date(), acquired.astimezone(timezone.utc).date())
            frames[name] = aggregate_pmms_monthly(scan_pmms_csv(sources["pmms_30y"]), as_of=as_of)
    summaries = {name: _validate(frame, table=name) for name, frame in frames.items()}
    output = root / "parquet"
    output.mkdir(parents=True, exist_ok=True)
    with _staging_directory(output) as temporary:
        for name, frame in frames.items():
            frame.sink_parquet(temporary / f"{name}.parquet", compression="zstd", row_group_size=250_000)
        for name in selected:
            (temporary / f"{name}.parquet").replace(output / f"{name}.parquet")
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("MACRO_DATA_ROOT"))
    parser.add_argument("--dataset", nargs="+", choices=list(SOURCE_DATASETS), help="Default: all three tables")
    args = parser.parse_args(argv)
    if args.data_root is None:
        parser.error("Set MACRO_DATA_ROOT or pass --data-root")
    root = args.data_root.expanduser().resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Data root must be outside the repository")
    try:
        summaries = build_parquet(root, datasets=args.dataset)
    except Exception as exc:
        print(f"[housing_macro] FAILED: {exc}", file=sys.stderr)
        return 1
    for name, summary in summaries.items():
        print(f"[{name}] {summary['rows']:,} rows, {summary['first_month']} to {summary['last_month']}, parquet/{name}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
