"""Build a vintage-stratified loan sample and its complete loan-month panel."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import time
import tomllib

import polars as pl

from quantbullet.data.freddie_sflld import (
    allocate_vintage_counts, filter_orig_loans, load_manifest, sample_loan_ids, scan_loan_panel,
)


@dataclass(frozen=True)
class SampleConfig:
    freddie_root: Path
    output_root: Path
    start_vintage: str
    end_vintage: str
    n_loans: int
    seed: int
    amortization_type: str | None = None
    original_loan_term: int | None = None


def _vintages(start: str, end: str) -> list[str]:
    if not all(isinstance(v, str) and re.fullmatch(r"\d{4}Q[1-4]", v) for v in (start, end)):
        raise ValueError("Vintage bounds must use YYYYQ1 through YYYYQ4")
    first, last = (int(v[:4]) * 4 + int(v[-1]) - 1 for v in (start, end))
    if first > last:
        raise ValueError("start_vintage must not exceed end_vintage")
    return [f"{index // 4:04d}Q{index % 4 + 1}" for index in range(first, last + 1)]


def read_config(path: str | Path) -> SampleConfig:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as file:
        config = tomllib.load(file)

    def local_path(value: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Data paths must be nonempty strings")

        def variable(match):
            name = match[1]
            if name not in os.environ or not os.environ[name]:
                raise ValueError(f"Set environment variable {name}")
            return os.environ[name]

        expanded = Path(re.sub(r"\$\{([A-Za-z_]\w*)\}", variable, value)).expanduser()
        return (expanded if expanded.is_absolute() else path.parent / expanded).resolve()

    sample = config["sample"]
    result = SampleConfig(
        local_path(config["data"]["freddie_root"]), local_path(config["data"]["output_root"]),
        sample["start_vintage"], sample["end_vintage"], sample["n_loans"], sample["seed"],
        sample.get("amortization_type"), sample.get("original_loan_term"),
    )
    _vintages(result.start_vintage, result.end_vintage)
    if type(result.n_loans) is not int or result.n_loans <= 0 or type(result.seed) is not int:
        raise ValueError("n_loans must be a positive integer and seed must be an integer")
    return result


def _source_paths(root: Path, vintages: list[str]) -> dict[str, dict[str, Path]]:
    quarters = load_manifest(root / "manifests/conversion.json")["quarters"]
    missing = sorted(set(vintages) - set(quarters))
    if missing:
        raise ValueError(f"Missing vintages: {', '.join(missing)}")
    sources = {}
    for vintage in vintages:
        sources[vintage] = {}
        for kind in ("orig", "perf"):
            relative = Path(quarters[vintage][kind]["path"])
            path = (root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"{vintage}: missing or invalid {kind} source path")
            sources[vintage][kind] = path
    return sources


def _clear_outputs(output: Path) -> None:
    """Remove only this process's files, after checking every resolved target."""
    targets = [output / name for name in (
        "sampled_loans.parquet", "sampling_summary.json", ".sampled_loans.parquet.tmp",
        ".sampling_summary.json.tmp",
    )]
    for directory in (output / "panel").glob("vintage=*"):
        if re.fullmatch(r"vintage=\d{4}Q[1-4]", directory.name):
            targets.extend(directory / name for name in ("panel.parquet", ".panel.parquet.tmp"))
    if any(not path.resolve().is_relative_to(output) for path in targets):
        raise ValueError("Output paths must remain inside the dedicated output directory")
    for path in targets:
        path.unlink(missing_ok=True)
    for directory in (output / "panel").glob("vintage=*"):
        if re.fullmatch(r"vintage=\d{4}Q[1-4]", directory.name) and not any(directory.iterdir()):
            directory.rmdir()


def _remove_temporary(path: Path) -> None:
    # Failed streaming writes can briefly hold a Windows file handle.
    for attempt in range(5):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))


def build_sample_panel(config: SampleConfig) -> dict:
    root, output = config.freddie_root.expanduser().resolve(), config.output_root.expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    if root.is_relative_to(repository) or output.is_relative_to(repository):
        raise ValueError("Input and output data directories must be outside the repository")
    vintages = _vintages(config.start_vintage, config.end_vintage)
    sources = _source_paths(root, vintages)
    if root.is_relative_to(output) or any(
        path.is_relative_to(output) for paths in sources.values() for path in paths.values()
    ):
        raise ValueError("Output must be a dedicated directory that does not contain source data")
    orig_sources = {
        vintage: pl.scan_parquet(paths["orig"], hive_partitioning=False)
        for vintage, paths in sources.items()
    }
    eligible_orig = {
        vintage: filter_orig_loans(
            orig, amortization_type=config.amortization_type, original_loan_term=config.original_loan_term,
        ) for vintage, orig in orig_sources.items()
    }
    raw_populations = {
        vintage: orig.select(pl.len()).collect().item() for vintage, orig in orig_sources.items()
    }
    populations = {vintage: orig.select(pl.len()).collect().item() for vintage, orig in eligible_orig.items()}
    print(f"[sample_panel] {sum(populations.values()):,} eligible orig loans; sampling {config.n_loans:,}", flush=True)
    quotas = allocate_vintage_counts(populations, config.n_loans)
    selected = []
    for vintage in vintages:
        ids = eligible_orig[vintage].select("loan_identifier").collect(engine="streaming")
        chosen = sample_loan_ids(ids, n_loans=quotas[vintage], seed=config.seed, vintage=vintage)
        selected.append(chosen.with_columns(pl.lit(vintage).alias("vintage")))
    sampled_ids = pl.concat(selected)
    if sampled_ids.height != config.n_loans or sampled_ids["loan_identifier"].n_unique() != config.n_loans:
        raise ValueError("Sample must contain exactly n_loans globally unique identifiers")

    output.mkdir(parents=True, exist_ok=True)
    _clear_outputs(output)
    static_frames, results = [], []
    for vintage in vintages:
        paths = sources[vintage]
        ids = sampled_ids.filter(pl.col("vintage") == vintage).select("loan_identifier")
        result = {
            "vintage": vintage, "population_loans": populations[vintage], "quota": quotas[vintage],
            "raw_population_loans": raw_populations[vintage],
            "sampled_loans": ids.height,
            "sampling_fraction": ids.height / populations[vintage] if populations[vintage] else 0.0,
            "orig_source": paths["orig"].relative_to(root).as_posix(),
            "perf_source": paths["perf"].relative_to(root).as_posix(),
            "panel_rows": 0, "panel_loans": 0, "loans_without_perf": ids.height,
        }
        if ids.height:
            print(f"[{vintage}] extracting {ids.height:,} sampled loans", flush=True)
            orig = pl.scan_parquet(paths["orig"], hive_partitioning=False).join(
                ids.lazy(), on="loan_identifier", how="semi",
            ).collect(engine="streaming").with_columns(pl.lit(vintage).alias("vintage"))
            if orig.height != ids.height:
                raise ValueError(f"{vintage}: sampled static rows do not match loan count")
            static_frames.append(orig)
            perf = pl.scan_parquet(paths["perf"], hive_partitioning=False).join(
                ids.lazy(), on="loan_identifier", how="semi",
            )
            expected_rows = perf.select(pl.len()).collect(engine="streaming").item()
            panel = scan_loan_panel(paths["orig"], paths["perf"], ids, vintage=vintage)
            # Check a small projection before writing; retain all original fields in the output.
            records = panel.select("loan_identifier", "period", "month").collect(engine="streaming")
            if records.height != expected_rows:
                raise ValueError(f"{vintage}: static join changed performance row count")
            if records.select(pl.struct("loan_identifier", "month").n_unique()).item() != records.height:
                raise ValueError(f"{vintage}: duplicate loan-month records")
            if records.join(ids, on="loan_identifier", how="anti").height:
                raise ValueError(f"{vintage}: panel contains loans outside the sample")
            result.update(
                panel_rows=records.height, panel_loans=records["loan_identifier"].n_unique(),
            )
            result["loans_without_perf"] = ids.height - result["panel_loans"]
            directory = output / "panel" / f"vintage={vintage}"
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / ".panel.parquet.tmp"
            try:
                panel.sink_parquet(temporary, compression="zstd", row_group_size=250_000)
                temporary.replace(directory / "panel.parquet")
            finally:
                _remove_temporary(temporary)
            print(f"[{vintage}] {result['panel_rows']:,} months; {result['loans_without_perf']:,} loans without perf", flush=True)
        results.append(result)

    summary = {
        "sampling": {
            "start_vintage": config.start_vintage, "end_vintage": config.end_vintage,
            "n_loans": config.n_loans, "seed": config.seed,
            "allocation": "proportional",
            "population": (
                "filtered_orig" if config.amortization_type is not None or config.original_loan_term is not None
                else "all_orig"
            ),
            "amortization_type": config.amortization_type, "original_loan_term": config.original_loan_term,
        },
        "population_loans": sum(populations.values()), "sampled_loans": sampled_ids.height,
        "raw_population_loans": sum(raw_populations.values()),
        "panel_rows": sum(row["panel_rows"] for row in results),
        "panel_loans": sum(row["panel_loans"] for row in results),
        "loans_without_perf": sum(row["loans_without_perf"] for row in results),
        "vintages": results,
    }
    temporary = output / ".sampled_loans.parquet.tmp"
    try:
        pl.concat(static_frames).sort("vintage", "loan_identifier").write_parquet(temporary, compression="zstd")
        temporary.replace(output / "sampled_loans.parquet")
    finally:
        _remove_temporary(temporary)
    temporary = output / ".sampling_summary.json.tmp"
    try:
        temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "sampling_summary.json")
    finally:
        _remove_temporary(temporary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = build_sample_panel(read_config(args.config))
    except Exception as exc:
        print(f"[sample_panel] FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        f"[sample_panel] {summary['sampled_loans']:,} sampled loans; "
        f"{summary['panel_loans']:,} panel loans; {summary['panel_rows']:,} loan-month rows",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
