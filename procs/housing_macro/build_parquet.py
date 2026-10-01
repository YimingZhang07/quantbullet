"""Normalize current housing macro snapshots into two versioned Parquet tables."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import uuid

import polars as pl

from quantbullet.data.fred import fred_csv_source, scan_cpi_csv
from quantbullet.data.zillow import scan_zhvi_csv, zhvi_sources


SCHEMA_VERSION = "housing_macro_monthly_v1"


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relative(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Manifest paths must be relative and remain inside the data root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Manifest path escapes the data root")
    return resolved


def _json_write(path: Path, value: dict) -> None:
    content = json.dumps(value, indent=2, ensure_ascii=False, default=lambda x: x.isoformat())
    path.write_text(content + "\n", encoding="utf-8")


def _inputs(root: Path) -> dict:
    manifest = json.loads((root / "manifests/downloads.json").read_text(encoding="utf-8"))
    inputs = {}
    for spec in (*zhvi_sources(), fred_csv_source("CPIAUCNS")):
        entry = manifest["datasets"][spec.dataset_id]
        current = entry["current"]
        metadata = current.get("metadata", entry.get("metadata"))
        if entry["provider"] != spec.provider or metadata != dict(spec.metadata):
            raise ValueError(f"{spec.dataset_id}: source metadata does not match the required dataset")
        path = _resolve_relative(root, current["path"])
        if (
            not path.is_file() or _sha256(path) != current["sha256"]
            or path.stat().st_size != current["bytes"]
        ):
            raise ValueError(f"{spec.dataset_id}: source file is missing or its hash/size does not match")
        inputs[spec.dataset_id] = {
            "path": current["path"], "sha256": current["sha256"], "bytes": current["bytes"],
            "provider": spec.provider, "metadata": dict(spec.metadata),
        }
    return inputs


def _validate(frame: pl.LazyFrame, *, hpi: bool) -> int:
    keys = ["provider", "metric", "geography_level", "region_id"] if hpi else ["series_id"]
    missing_id = pl.any_horizontal(*[
        pl.col(key).is_null() | (pl.col(key).str.strip_chars() == "") for key in keys
    ])
    summary = frame.select(
        pl.len().alias("rows"),
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
    return summary["rows"]


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
        f"Schema: `{SCHEMA_VERSION}`. Values retain original units (ZHVI dollars; CPI index).", "",
        "Nulls are retained. Missing dates are absent source columns/records; null values "
        "are present cells without a value. Region counts include regions with no valid "
        "values. Full per-month and per-region details are in coverage.json.", "",
    ]
    for table in ("hpi", "cpi"):
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


def _artifact(path: Path, relative: str, *, rows: int | None = None) -> dict:
    value = {"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size}
    if rows is not None:
        value["rows"] = rows
    return value


def build_parquet(data_root: str | Path) -> dict:
    """Build from verified current snapshots; publish the manifest only after all outputs succeed.

    Use one writer per data root. The returned build entry has status built or cached.
    """
    root = Path(data_root).expanduser().resolve()
    inputs = _inputs(root)
    manifest_path = root / "manifests/normalization.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {"manifest_version": 1, "versions": []}
    )
    previous = manifest.get("current", {})
    if previous.get("inputs") == inputs and previous.get("schema_version") == SCHEMA_VERSION:
        artifacts = list(previous.get("outputs", {}).values()) + list(previous.get("reports", {}).values())
        if len(artifacts) == 4 and all(
            (path := _resolve_relative(root, artifact["path"])).is_file()
            and path.stat().st_size == artifact["bytes"] and _sha256(path) == artifact["sha256"]
            for artifact in artifacts
        ):
            return {**previous, "status": "cached"}
    now = datetime.now(timezone.utc)
    build_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
    scratch = root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with _staging_directory(scratch) as stage:
        print("[housing_macro] converting verified snapshots", flush=True)
        hpi = pl.concat([
            scan_zhvi_csv(_resolve_relative(root, inputs[f"zhvi_{geography}"]["path"]), geography=geography)
            for geography in ("metro", "state", "zip")
        ])
        cpi = scan_cpi_csv(_resolve_relative(root, inputs["cpi"]["path"]))
        hpi.sink_parquet(stage / "hpi.parquet", compression="zstd", row_group_size=250_000)
        cpi.sink_parquet(stage / "cpi.parquet", compression="zstd")
        hpi, cpi = pl.scan_parquet(stage / "hpi.parquet"), pl.scan_parquet(stage / "cpi.parquet")
        counts = {"hpi": _validate(hpi, hpi=True), "cpi": _validate(cpi, hpi=False)}
        print(f"[housing_macro] validated HPI={counts['hpi']:,} CPI={counts['cpi']:,}; computing coverage", flush=True)
        report = {
            "schema_version": SCHEMA_VERSION, "inputs": inputs,
            "hpi": _coverage(hpi, hpi=True), "cpi": _coverage(cpi, hpi=False),
        }
        _json_write(stage / "coverage.json", report)
        (stage / "coverage.md").write_text(_markdown(report), encoding="utf-8")
        entry = {
            "build_id": build_id, "created_at_utc": now.isoformat(),
            "schema_version": SCHEMA_VERSION, "inputs": inputs, "outputs": {}, "reports": {},
        }
        for name in ("hpi", "cpi"):
            relative = f"parquet/{build_id}/{name}.parquet"
            entry["outputs"][name] = _artifact(stage / f"{name}.parquet", relative, rows=counts[name])
        for name in ("coverage.json", "coverage.md"):
            relative = f"reports/{build_id}/{name}"
            entry["reports"][name] = _artifact(stage / name, relative)
        # All four artifacts are complete before publishing anything. The manifest is the commit point.
        for artifact in (*entry["outputs"].values(), *entry["reports"].values()):
            destination = _resolve_relative(root, artifact["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            (stage / destination.name).replace(destination)
        manifest["current"] = entry
        manifest["versions"].append(entry)
        staged_manifest = stage / "normalization.json"
        _json_write(staged_manifest, manifest)
        staged_manifest.replace(manifest_path)
    return {**entry, "status": "built"}


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
        result = build_parquet(root)
    except Exception as exc:
        print(f"[housing_macro] FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"[housing_macro] {result['status']} build={result['build_id']}", flush=True)
    for name, artifact in result["outputs"].items():
        print(f"[{name}] {artifact['rows']:,} rows, {artifact['bytes']:,} bytes, {artifact['path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
