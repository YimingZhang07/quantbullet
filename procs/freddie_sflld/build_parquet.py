"""Convert downloaded Freddie Mac SFLLD Standard data by origination vintage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from quantbullet.data.freddie_sflld import SFLLDArchive, convert_vintage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("FREDDIE_DATA_ROOT"))
    parser.add_argument("--source", type=Path, help="Override the full Standard Dataset ZIP path")
    parser.add_argument("--quarter", help="Convert only this vintage, e.g. 2015Q1")
    parser.add_argument("--start", default="2015Q1", help="First vintage (inclusive)")
    parser.add_argument("--end", help="Last vintage (inclusive); default: latest available")
    args = parser.parse_args(argv)
    if args.data_root is None:
        parser.error("Set FREDDIE_DATA_ROOT or pass --data-root")
    root = args.data_root.expanduser().resolve()
    source = args.source or root / "raw" / "full_set_standard_historical_data.zip"
    if not source.is_file():
        parser.error(f"Source archive not found: {source}")
    archive = SFLLDArchive(source)
    vintages = archive.list_vintages()
    if args.quarter:
        if args.end or args.start != "2015Q1":
            parser.error("--quarter cannot be combined with --start or --end")
        selected = [args.quarter]
    else:
        selected = [v for v in vintages if v >= args.start and (args.end is None or v <= args.end)]
    missing = sorted(set(selected) - set(vintages))
    if missing or not selected:
        parser.error(f"No matching vintages; missing: {missing}")
    print(f"Converting {len(selected)} vintage(s): {selected[0]} to {selected[-1]}", flush=True)
    failures = []
    for vintage in selected:
        print(f"[{vintage}] checking source", flush=True)
        try:
            result = convert_vintage(archive, vintage, root)
        except Exception as exc:
            failures.append(vintage)
            print(f"[{vintage}] FAILED: {exc}", file=sys.stderr, flush=True)
            continue
        print(
            f"[{vintage}] {result['status']} "
            f"orig={result['orig']['rows']:,} perf={result['perf']['rows']:,}",
            flush=True,
        )
    if failures:
        print(f"Failed vintages: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
