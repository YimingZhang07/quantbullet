"""Clean sampled Freddie panels and derive generic Polars features and states."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import tomllib
from uuid import uuid4

import polars as pl

from quantbullet.data.freddie_sflld.features import (
    FEATURE_COLUMNS, NUMERIC_FEATURES, ORIG_NUMBERS, QUALITY_FLAGS,
    derive_loan_features, prepare_loan_months, prepare_macro_tables, validate_panel,
)


# Only these generated files belong to this process's output directory.
OWNED_OUTPUTS = {"panel.parquet", "preparation_summary.json"}


@dataclass(frozen=True)
class PreparationConfig:
    sample_root: Path
    macro_root: Path
    output_root: Path
    burnout_threshold: float = 0.5

    def __post_init__(self):
        value = self.burnout_threshold
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("burnout_threshold must be a finite nonnegative number")


def read_config(path: str | Path) -> PreparationConfig:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as file:
        config = tomllib.load(file)
        data = config["data"]

    def local_path(value: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Data paths must be nonempty strings")

        def variable(match):
            if not os.environ.get(match[1]):
                raise ValueError(f"Set environment variable {match[1]}")
            return os.environ[match[1]]

        expanded = Path(re.sub(r"\$\{([A-Za-z_]\w*)\}", variable, value)).expanduser()
        return (expanded if expanded.is_absolute() else path.parent / expanded).resolve()

    return PreparationConfig(
        *(local_path(data[name]) for name in ("sample_root", "macro_root", "output_root")),
        burnout_threshold=config.get("features", {}).get("burnout_threshold", 0.5),
    )


def _check_siblings(paths: list[Path], parent: Path) -> None:
    # Check absolute targets before directory moves or recursive cleanup.
    if any(path.resolve().parent != parent.resolve() or path.is_symlink() for path in paths):
        raise ValueError("Publication and cleanup targets must remain in the output parent")


def _remove_directory(path: Path, parent: Path) -> None:
    _check_siblings([path], parent)
    for attempt in range(5):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def _publish(staging: Path, output: Path) -> None:
    """Replace an exclusively owned output directory, restoring on rename failure."""
    backup = output.parent / f".{output.name}-previous-{uuid4().hex}"
    _check_siblings([staging, output, backup], output.parent)
    if output.exists():
        if not output.is_dir() or set(p.name for p in output.iterdir()) - OWNED_OUTPUTS:
            raise ValueError("Output directory must contain only this process's panel and summary")
        output.rename(backup)
    try:
        staging.rename(output)
    except Exception:
        if backup.exists():
            _check_siblings([backup, output], output.parent)
            backup.rename(output)
        raise
    if backup.exists():
        _remove_directory(backup, output.parent)


def _merge_monthly(paths: list[Path], target: Path) -> None:
    """Stream a balanced merge of already month-sorted temporary quarter files."""
    frames = [pl.scan_parquet(path, hive_partitioning=False) for path in paths]
    while len(frames) > 1:
        frames = [
            frames[index].merge_sorted(frames[index + 1], key="d_reporting_month")
            if index + 1 < len(frames) else frames[index]
            for index in range(0, len(frames), 2)
        ]
    frames[0].sink_parquet(target, compression="zstd", row_group_size=250_000, engine="streaming")


def _partition_summary(path: Path, expected_rows: int, sampled_loans: int) -> dict:
    frame = pl.scan_parquet(path, hive_partitioning=False)
    unknown = pl.col("f_pre_status").is_null() | (pl.col("f_pre_status") == "UNKNOWN")
    invalid_age = pl.col("c_age").is_null() | (pl.col("c_age") < 0)
    stats = frame.select(
        pl.len().alias("rows"),
        pl.col("loan_identifier").n_unique().alias("loans"),
        pl.col("c_prev_balance").is_null().sum().alias("missing_previous_balance"),
        (pl.col("c_prev_balance") <= 0).sum().alias("nonpositive_previous_balance"),
        unknown.sum().alias("unknown_previous_status"),
        invalid_age.sum().alias("invalid_age"),
        *[pl.col(name).sum().alias(name) for name in QUALITY_FLAGS],
        pl.col("is_ever_modified").sum().alias("ever_modified_rows"),
        (pl.col("f_hpi_level") == "national").sum().alias("national_hpi_rows"),
        pl.col("d_reporting_month").min().alias("first_month"),
        pl.col("d_reporting_month").max().alias("last_month"),
        pl.any_horizontal(*(
            ~pl.col(name).is_finite().fill_null(True) for name in NUMERIC_FEATURES
        )).sum().alias("nonfinite_features"),
        *[pl.col(name).is_null().sum().alias(f"null_{name}") for name in FEATURE_COLUMNS],
    ).collect(engine="streaming").row(0, named=True)
    if stats["rows"] != expected_rows:
        raise ValueError("Feature joins changed the panel row count")
    if stats.pop("nonfinite_features"):
        raise ValueError("Prepared panel contains nonfinite features")
    stats["sampled_loans"] = sampled_loans
    stats["loans_without_perf"] = sampled_loans - stats["loans"]
    stats["feature_missing"] = {
        name: {"rows": stats.pop(f"null_{name}"), "fraction": None} for name in FEATURE_COLUMNS
    }
    for counts in stats["feature_missing"].values():
        counts["fraction"] = counts["rows"] / stats["rows"] if stats["rows"] else None
    reasons = (
        "missing_previous_balance", "nonpositive_previous_balance", "unknown_previous_status",
        "invalid_age", *QUALITY_FLAGS,
    )
    stats["quality_counts"] = {key: stats.pop(key) for key in reasons}
    stats["quality_counts_overlap"] = True
    stats["status_rows"] = dict(frame.group_by("f_status").len().collect(engine="streaming").iter_rows())
    stats["exit_reason_rows"] = dict(
        frame.filter(pl.col("f_exit_reason").is_not_null()).group_by("f_exit_reason")
        .len().collect(engine="streaming").iter_rows()
    )
    stats["national_hpi_loans"] = frame.filter(pl.col("f_hpi_level") == "national").select(
        pl.col("loan_identifier").n_unique()
    ).collect(engine="streaming").item()
    return stats


def build_prepared_panel(config: PreparationConfig, *, vintages: list[str] | None = None) -> dict:
    sample, macro_root, output = (p.expanduser().resolve() for p in (
        config.sample_root, config.macro_root, config.output_root,
    ))
    repository = Path(__file__).resolve().parents[2]
    if any(path.is_relative_to(repository) for path in (sample, macro_root, output)):
        raise ValueError("Input and output data directories must be outside the repository")
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in (sample, macro_root)):
        raise ValueError("Output must be separate from input directories")
    if output.exists() and (
        not output.is_dir() or set(p.name for p in output.iterdir()) - OWNED_OUTPUTS
    ):
        raise ValueError("Output directory must contain only this process's panel and summary")
    with (sample / "sampling_summary.json").open(encoding="utf-8") as file:
        sampling = json.load(file)
    entries = {row["vintage"]: row for row in sampling["vintages"]}
    selected = sorted(set(vintages) if vintages is not None else entries)
    if not selected or any(not re.fullmatch(r"\d{4}Q[1-4]", v) or v not in entries for v in selected):
        raise ValueError("Requested vintages must exist in sampling_summary.json")
    paths = {}
    for vintage in selected:
        if entries[vintage]["sampled_loans"]:
            path = sample / "panel" / f"vintage={vintage}" / "panel.parquet"
            if not path.is_file() or not path.resolve().is_relative_to(sample):
                raise ValueError(f"{vintage}: missing or invalid panel partition")
            paths[vintage] = path
    if not paths:
        raise ValueError("Selected vintages contain no sampled loans")
    hpi = pl.scan_parquet(macro_root / "parquet/hpi.parquet").filter(
        pl.col("geography_level").is_in(["state", "national"])
    ).collect(engine="streaming")
    macro = prepare_macro_tables(
        hpi, pl.read_parquet(macro_root / "parquet/pmms.parquet"),
        pl.read_parquet(macro_root / "parquet/cpi.parquet"),
    )
    loans = derive_loan_features(pl.scan_parquet(sample / "sampled_loans.parquet"), macro=macro).collect(
        engine="streaming",
    )
    ids = loans["loan_identifier"]
    if ids.null_count() or ids.str.strip_chars().eq("").any() or ids.n_unique() != loans.height:
        raise ValueError("Sampled loans must have globally unique nonempty identifiers")
    if loans.height != sampling["sampled_loans"]:
        raise ValueError("Sampled loan count does not match sampling summary")
    if loans.select(pl.any_horizontal(*[
        ~pl.col(name).is_finite().fill_null(True)
        for name in [*(name for name, _ in ORIG_NUMBERS.values()), "c_orig_pmms", "c_sato"]
    ]).any()).item():
        raise ValueError("Sampled loans contain nonfinite numeric features")
    if loans["f_vintage"].null_count() or set(loans["f_vintage"].unique()) != {
        v for v, entry in entries.items() if entry["sampled_loans"]
    }:
        raise ValueError("Sampled loan vintages do not match sampling summary")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-prepare-", dir=output.parent)).resolve()
    results = []
    try:
        quarters = staging / "_quarters"
        quarters.mkdir()
        prepared_paths = []
        for vintage in selected:
            chosen = loans.filter(pl.col("f_vintage") == vintage)
            if chosen.height != entries[vintage]["sampled_loans"]:
                raise ValueError(f"{vintage}: sampled loan count does not match summary")
            if vintage not in paths:
                continue
            print(f"[{vintage}] preparing {chosen.height:,} loans", flush=True)
            panel = pl.scan_parquet(paths[vintage], hive_partitioning=False)
            expected_rows = validate_panel(panel)
            if expected_rows != entries[vintage]["panel_rows"]:
                raise ValueError(f"{vintage}: panel row count does not match sampling summary")
            if panel.select("loan_identifier").unique().join(
                chosen.select("loan_identifier").lazy(), on="loan_identifier", how="anti",
            ).limit(1).collect(engine="streaming").height:
                raise ValueError(f"{vintage}: panel loan is outside the sample")
            if panel.filter(pl.col("vintage").is_null() | (pl.col("vintage") != vintage)).limit(1).collect(
                engine="streaming",
            ).height:
                raise ValueError(f"{vintage}: mismatched panel vintage")
            target = quarters / f"{vintage}.parquet"
            prepare_loan_months(
                panel, chosen.lazy(), macro=macro, burnout_threshold=config.burnout_threshold,
            ).sort("d_reporting_month").sink_parquet(
                target, compression="zstd", row_group_size=250_000,
            )
            prepared_paths.append(target)
            result = _partition_summary(target, expected_rows, chosen.height)
            result.update(vintage=vintage)
            results.append(result)
            print(
                f"[{vintage}] {result['rows']:,} rows; {result['loans']:,} loans", flush=True,
            )
        print("[prepare_panel] merging prepared quarters into month-sorted panel.parquet", flush=True)
        target = staging / "panel.parquet"
        _merge_monthly(prepared_paths, target)
        if pl.scan_parquet(target).select(pl.len()).collect().item() != sum(r["rows"] for r in results):
            raise ValueError("Final merge changed the panel row count")
        _remove_directory(quarters, staging)
        summary = {
            "path": "panel.parquet", "sorted_by": ["d_reporting_month"],
            "vintages": results, "rows": sum(r["rows"] for r in results),
            "sampled_loans": sum(entries[v]["sampled_loans"] for v in selected),
            "panel_loans": sum(r["loans"] for r in results),
            "national_hpi_rows": sum(r["national_hpi_rows"] for r in results),
            "ever_modified_rows": sum(r["ever_modified_rows"] for r in results),
            "quality_counts": {
                key: sum(r["quality_counts"][key] for r in results)
                for key in results[0]["quality_counts"]
            },
            "feature_columns": list(FEATURE_COLUMNS),
            "burnout_threshold": config.burnout_threshold,
            "conventions": {
                "origination_month": "first payment month minus one month; approximate",
                "age": "months since inferred origination; not reset by modification",
                "macro_lag": "one observation month; not a historical publication-time guarantee",
                "hpi": "state pair when available, otherwise national pair; raw ZHVI dollars",
                "hpi_ratio": "lag1 ZHVI divided by origination ZHVI; 1.0 means unchanged",
                "status": "reported row status; distinct exit reasons; 01 is voluntary payoff",
                "quality_flags": "descriptive only; no row filtering or target assignment",
                "post_exit": "after earliest reported/effective known exit month; retrospective flag",
                "rates": "percent; spreads in percentage points",
                "payments": "original-contract P&I estimate; prior actual balance/rate for monthly split; USD",
                "ever_modified": "observed Y/P through the current row; true from first modification onward",
                "burnout": "original-rate exposure from inferred origination through reporting month minus 2; calendar months; missing PMMS makes exposure null",
                "modeling": "select predictors, define targets and risk sets downstream",
            },
        }
        summary["loans_without_perf"] = summary["sampled_loans"] - summary["panel_loans"]
        (staging / "preparation_summary.json").write_text(
            json.dumps(summary, indent=2, default=lambda value: value.isoformat()) + "\n", encoding="utf-8",
        )
        _publish(staging, output)
        return summary
    finally:
        if staging.exists():
            _remove_directory(staging, output.parent)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vintage", nargs="+", help="Build only selected sampled vintages (e.g. 2015Q1)")
    args = parser.parse_args(argv)
    try:
        summary = build_prepared_panel(read_config(args.config), vintages=args.vintage)
    except Exception as exc:
        print(f"[prepare_panel] FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        f"[prepare_panel] {summary['rows']:,} rows; {summary['panel_loans']:,} loans", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
