"""Inspect HPI/CPI tables and, when present, monthly PMMS coverage."""

from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import sys

import polars as pl


def _month_number(month: date) -> int:
    return month.year * 12 + month.month - 1


def _months(start: date, end: date) -> list[date]:
    return [
        date(number // 12, number % 12 + 1, 1)
        for number in range(_month_number(start), _month_number(end) + 1)
    ]


def _coverage_window(frame: pl.LazyFrame, registry: pl.DataFrame, start: date, end: date) -> dict:
    keys = ["geography_level", "region_id"]
    window = frame.filter(pl.col("month").is_between(start, end))
    groups = window.group_by(keys).agg(
        pl.len().alias("rows"), pl.col("value").count().alias("valid_values"),
        pl.col("value").null_count().alias("null_values"),
        (pl.col("value") <= 0).sum().alias("nonpositive_values"),
        pl.col("month").min().alias("first_present_month"), pl.col("month").max().alias("last_present_month"),
        pl.col("month").filter(pl.col("value").is_not_null()).min().alias("first_valid_month"),
        pl.col("month").filter(pl.col("value").is_not_null()).max().alias("last_valid_month"),
    ).collect(engine="streaming")
    bounded = window.join(groups.lazy().select(*keys, "first_valid_month", "last_valid_month"), on=keys).filter(
        pl.col("month").is_between(pl.col("first_valid_month"), pl.col("last_valid_month"))
    ).group_by(keys).agg(
        pl.len().alias("present_months_between_valid_bounds"),
        pl.col("value").null_count().alias("null_months_between_valid_bounds"),
    ).collect(engine="streaming")
    regions = (
        registry.join(groups, on=keys, how="left")
        .join(bounded, on=keys, how="left").sort(keys).to_dicts()
    )
    calendar = _months(start, end)
    summaries = {}
    for row in regions:
        for name in ("rows", "valid_values", "null_values", "nonpositive_values"):
            row[name] = row[name] or 0
        row["missing_date_months"] = len(calendar) - row["rows"]
        row["null_fraction"] = row["null_values"] / row["rows"] if row["rows"] else None
        first, last = row["first_valid_month"], row["last_valid_month"]
        span = _month_number(last) - _month_number(first) + 1 if first else None
        row["missing_date_months_between_valid_bounds"] = (
            span - row["present_months_between_valid_bounds"] if span else None
        )
        row["missing_values_between_valid_bounds"] = span - row["valid_values"] if span else None
        level = row["geography_level"]
        summary = summaries.setdefault(level, {
            "geography_level": level, "regions": 0, "rows": 0, "valid_values": 0,
            "null_values": 0, "missing_date_months": 0, "nonpositive_values": 0,
            "first_present_month": None, "last_present_month": None,
        })
        summary["regions"] += 1
        for name in ("rows", "valid_values", "null_values", "missing_date_months", "nonpositive_values"):
            summary[name] += row[name]
        for name, chooser in (("first_present_month", min), ("last_present_month", max)):
            if row[name] is not None:
                summary[name] = chooser(summary[name], row[name]) if summary[name] else row[name]
    for summary in summaries.values():
        summary["null_fraction"] = summary["null_values"] / summary["rows"] if summary["rows"] else None
        summary["expected_region_months"] = summary["regions"] * len(calendar)
    monthly_raw = window.group_by("geography_level", "month").agg(
        pl.len().alias("present_regions"), pl.col("value").count().alias("valid_regions"),
        pl.col("value").null_count().alias("null_regions"),
        (pl.col("value") <= 0).sum().alias("nonpositive_regions"),
    ).collect(engine="streaming").to_dicts()
    lookup = {(row["geography_level"], row["month"]): row for row in monthly_raw}
    monthly = []
    for level, summary in sorted(summaries.items()):
        for month in calendar:
            row = lookup.get((level, month), {
                "geography_level": level, "month": month, "present_regions": 0,
                "valid_regions": 0, "null_regions": 0, "nonpositive_regions": 0,
            })
            row["missing_date_regions"] = summary["regions"] - row["present_regions"]
            monthly.append(row)
    return {
        "start_month": start, "end_month": end,
        "summary": list(summaries.values()), "monthly": monthly, "regions": regions,
    }


def _coverage(frame: pl.LazyFrame, *, hpi: bool) -> dict:
    if not hpi:
        frame = frame.select(
            pl.lit("national").alias("geography_level"),
            pl.col("series_id").alias("region_id"), pl.col("series_id").alias("region_name"),
            "month", "value",
        )
    registry = frame.select("geography_level", "region_id", "region_name").unique().collect(engine="streaming")
    # Names are metadata, not part of the region key; inconsistent names are rejected.
    if registry.select(pl.struct("geography_level", "region_id").n_unique()).item() != registry.height:
        raise ValueError("Inconsistent region names for the same region ID")
    start, end = frame.select(
        pl.col("month").min(), pl.col("month").max().alias("last"),
    ).collect().row(0)
    return {
        "full_history": _coverage_window(frame, registry, start, end),
        "from_2015": _coverage_window(frame, registry, max(start, date(2015, 1, 1)), end),
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Housing macro coverage", "",
        "Values retain original units (ZHVI dollars; CPI index; PMMS percent).", "",
        "Nulls are retained. Missing dates are absent source columns/records; null values "
        "are present cells without a value. Region counts include regions with no valid "
        "values. Full per-month and per-region details are in coverage.json.", "",
    ]
    for table in report:
        for period in ("full_history", "from_2015"):
            window = report[table][period]
            lines += [
                f"## {table.upper()} — {period}", "",
                f"Window: {window['start_month']} through {window['end_month']}", "",
                "| Level | Regions | Present month range | Rows | Valid | Null | Null % | Missing dates | Nonpositive |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
            for row in window["summary"]:
                fraction = f"{row['null_fraction']:.2%}" if row["null_fraction"] is not None else "n/a"
                lines.append(
                    f"| {row['geography_level']} | {row['regions']:,} "
                    f"| {row['first_present_month']}–{row['last_present_month']} "
                    f"| {row['rows']:,} | {row['valid_values']:,} | {row['null_values']:,} "
                    f"| {fraction} | {row['missing_date_months']:,} | {row['nonpositive_values']:,} |"
                )
            lines.append("")
    cpi_gaps = [
        f"{row['month']} (null={row['null_regions']}, missing date={row['missing_date_regions']})"
        for row in report["cpi"]["full_history"]["monthly"]
        if row["null_regions"] or row["missing_date_regions"]
    ]
    lines += ["## CPI gaps", "", *([f"- {gap}" for gap in cpi_gaps] or ["None."]), ""]
    return "\n".join(lines)


def write_coverage(data_root: str | Path) -> dict:
    """Read existing Parquet tables; write full-history and 2015-onward coverage reports."""
    root = Path(data_root).expanduser().resolve()
    tables = ["hpi", "cpi"]
    if (root / "parquet/pmms.parquet").is_file():
        tables.append("pmms")
    report = {
        name: _coverage(pl.scan_parquet(root / "parquet" / f"{name}.parquet"), hpi=name == "hpi")
        for name in tables
    }
    output = root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    (output / "coverage.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=lambda value: value.isoformat()) + "\n",
        encoding="utf-8",
    )
    (output / "coverage.md").write_text(_markdown(report), encoding="utf-8")
    return report


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
        write_coverage(root)
    except Exception as exc:
        print(f"[coverage] FAILED: {exc}", file=sys.stderr)
        return 1
    print("[coverage] wrote reports/coverage.json and reports/coverage.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
