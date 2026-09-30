"""Convert SFLLD Standard Dataset quarters to separate Parquet datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import tempfile
import uuid

import polars as pl

from .archive import SFLLDArchive
from .schema import ORIG_COLUMNS, PERF_COLUMNS, SCHEMA_VERSION


MANIFEST_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"manifest_version": MANIFEST_VERSION, "quarters": {}}
    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version in {path}")
    return manifest


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_text(path: Path, expected_columns: int) -> None:
    with path.open("rb") as file:
        for index in range(20):
            line = file.readline()
            if not line:
                if index == 0:
                    raise ValueError(f"Empty data file: {path.name}")
                return
            columns = line.rstrip(b"\r\n").count(b"|") + 1
            if columns != expected_columns:
                raise ValueError(
                    f"{path.name} line {index + 1}: expected {expected_columns} "
                    f"columns, got {columns}"
                )


def _write_parquet(text_path: Path, output_path: Path, columns: tuple[str, ...]) -> None:
    _validate_text(text_path, len(columns))
    frame = pl.scan_csv(
        text_path,
        has_header=False,
        separator="|",
        quote_char=None,
        infer_schema=False,
        new_columns=columns,
        empty_string_is_null=True,
    )
    frame.sink_parquet(output_path, compression="zstd", row_group_size=250_000)


def _check_parquet(path: Path, columns: tuple[str, ...], kind: str) -> int:
    frame = pl.scan_parquet(path)
    if tuple(frame.collect_schema().names()) != columns:
        raise ValueError(f"Unexpected {kind} Parquet schema: {path}")
    checks = frame.select(
        pl.len().alias("rows"),
        pl.col("loan_identifier").null_count().alias("null_loan_ids"),
        *([pl.col("period").null_count().alias("null_periods")] if kind == "perf" else []),
    ).collect(engine="streaming").row(0, named=True)
    if checks["rows"] == 0 or checks["null_loan_ids"]:
        raise ValueError(f"Empty {kind} data or null Loan Identifier: {path}")
    if kind == "perf" and checks["null_periods"]:
        raise ValueError(f"Null reporting period: {path}")
    return checks["rows"]


def _unchanged(entry: dict | None, source_hash: str, root: Path) -> bool:
    if not entry or entry.get("source_sha256") != source_hash:
        return False
    if entry.get("schema_version") != SCHEMA_VERSION:
        return False
    for kind in ("orig", "perf"):
        output = entry.get(kind, {})
        relative_path = output.get("path")
        expected_hash = output.get("sha256")
        if not relative_path or not expected_hash:
            return False
        path = root / relative_path
        if not path.is_file() or file_sha256(path) != expected_hash:
            return False
    return True


def convert_vintage(
    archive: SFLLDArchive,
    vintage: str,
    data_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> dict:
    """Convert one vintage, commit both outputs, then atomically update manifest.

    Versioned Parquet filenames keep previous successful outputs intact if a
    conversion fails. Readers should use the manifest to select current files.
    """
    root = Path(data_root)
    manifest_file = Path(manifest_path) if manifest_path else root / "manifests" / "conversion.json"
    manifest = load_manifest(manifest_file)
    source_hash = archive.quarter_hash(vintage)
    previous = manifest["quarters"].get(vintage)
    if _unchanged(previous, source_hash, root):
        return {"vintage": vintage, **previous, "status": "skipped"}

    scratch = root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{vintage}-", dir=scratch) as temporary:
        workdir = Path(temporary)
        _, text_paths = archive.extract_quarter(vintage, workdir)
        staged = {kind: workdir / f"{kind}.parquet" for kind in ("orig", "perf")}
        rows = {}
        for kind, columns in (("orig", ORIG_COLUMNS), ("perf", PERF_COLUMNS)):
            _write_parquet(text_paths[kind], staged[kind], columns)
            rows[kind] = _check_parquet(staged[kind], columns, kind)
            text_paths[kind].unlink()

        outputs = {}
        run_id = uuid.uuid4().hex[:12]
        for kind in ("orig", "perf"):
            relative = (
                Path("parquet") / kind / f"vintage={vintage}"
                / f"data-{source_hash[:16]}-{run_id}.parquet"
            )
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged[kind], target)
            outputs[kind] = {
                "path": relative.as_posix(),
                "rows": rows[kind],
                "sha256": file_sha256(target),
                "bytes": target.stat().st_size,
            }

    entry = {
        "source_archive": archive.source.name,
        "source_member": f"historical_data_{vintage[:4]}.zip/historical_data_{vintage}.zip",
        "source_sha256": source_hash,
        "schema_version": SCHEMA_VERSION,
        "converted_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        **outputs,
    }
    manifest["quarters"][vintage] = entry
    _write_manifest(manifest_file, manifest)

    # A previous version is no longer selected by the manifest. Do not remove
    # any newer output if a caller happens to rerun the same source.
    if previous:
        for kind in ("orig", "perf"):
            old_relative = previous.get(kind, {}).get("path")
            if old_relative and old_relative != outputs[kind]["path"]:
                old_path = root / old_relative
                old_path.unlink(missing_ok=True)
    return {"vintage": vintage, **entry, "status": "converted"}
