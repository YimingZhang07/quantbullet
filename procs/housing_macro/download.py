"""Download Zillow ZHVI, national CPI, and weekly PMMS into an external data root."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from quantbullet.data.download import download_source
from quantbullet.data.fred import fred_csv_source
from quantbullet.data.zillow import zhvi_sources


def _progress(dataset_id: str):
    last_bytes = 0
    last_time = time.monotonic()

    def report(size: int, total: int | None) -> None:
        nonlocal last_bytes, last_time
        now = time.monotonic()
        if size < last_bytes or size - last_bytes >= 16 * 1024 * 1024 or now - last_time >= 10:
            suffix = f" / {total / 1024**2:.1f} MiB" if total is not None else ""
            print(f"[{dataset_id}] {size / 1024**2:.1f} MiB{suffix}", flush=True)
            last_bytes, last_time = size, now

    return report


def main(argv: list[str] | None = None) -> int:
    sources = {spec.dataset_id: spec for spec in (
        *zhvi_sources(), fred_csv_source("CPIAUCNS"), fred_csv_source("MORTGAGE30US"),
    )}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("MACRO_DATA_ROOT"))
    parser.add_argument("--dataset", nargs="+", choices=list(sources), help="Default: all five datasets")
    parser.add_argument("--refresh", action="store_true", help="Fetch remote files even when cached")
    args = parser.parse_args(argv)
    if args.data_root is None:
        parser.error("Set MACRO_DATA_ROOT or pass --data-root")
    root = args.data_root.expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    if root.is_relative_to(repository):
        parser.error("Data root must be outside the repository")
    selected = list(dict.fromkeys(args.dataset or sources))
    failed = []
    for dataset_id in selected:
        print(f"[{dataset_id}] checking source", flush=True)
        try:
            result = download_source(
                sources[dataset_id], root, refresh=args.refresh, progress=_progress(dataset_id),
            )
        except Exception as exc:
            failed.append(dataset_id)
            print(f"[{dataset_id}] FAILED: {exc}", file=sys.stderr, flush=True)
            continue
        print(f"[{dataset_id}] {result.status} {result.bytes:,} bytes sha256={result.sha256}", flush=True)
    if failed:
        print(f"Failed datasets: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
