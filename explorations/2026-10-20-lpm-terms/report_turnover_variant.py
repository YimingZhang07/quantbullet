"""Fit and report one turnover variant in an output root of its own; the production artifacts are not touched.

The variant is the basis of the numeric features (``--basis``: the production FlatRamp spec, or the tent spec of
compare_tent_turnover.py with its references, or none with ``--mean-one``) and the config's form (``block_link``
and its solver settings). The prepared frame of the config's output_root is linked (or copied) into --out, which
must be outside the repository; the turnover fit and report then run there.

Run from the repository root (one line):
    .venv/Scripts/python.exe explorations/2026-10-20-lpm-terms/report_turnover_variant.py
        --config projects/freddie_prepayment/turnover.example.toml --out <directory> --basis tent [--mean-one]
"""
import argparse
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import compare_tent_turnover as compare  # noqa: E402  (this folder is first on sys.path when run as a script)
from projects.freddie_prepayment import fit_turnover, report_turnover  # noqa: E402
from projects.freddie_prepayment.config import REPOSITORY, read_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="Output root for this variant's fit and report")
    parser.add_argument("--basis", choices=("tent", "flat_ramp"), required=True)
    parser.add_argument("--mean-one", action="store_true", help="Tent without references: every block keeps mean 1")
    args = parser.parse_args()
    if args.mean_one and args.basis != "tent":
        raise ValueError("--mean-one applies to --basis tent; the FlatRamp spec has no references")
    config = read_config(args.config)
    out = Path(args.out).resolve()
    if out.is_relative_to(REPOSITORY):
        raise ValueError("--out must be outside the repository")
    out.mkdir(parents=True, exist_ok=True)
    source, target = config.output_root / "turnover_frame.parquet", out / "turnover_frame.parquet"
    if not target.exists():
        try:
            os.link(source, target)        # same volume: no second copy of the frame
        except OSError:
            shutil.copy2(source, target)
    variant = replace(config, output_root=out)
    if args.basis == "tent":
        references = {} if args.mean_one else compare.REFERENCES
        fit_turnover.build_spec = lambda: compare.tent_spec(references)   # fit() builds its spec through this name
    fit_turnover.fit(variant)
    report_turnover.report(variant)
    print(f"[variant] report written to {out / 'turnover_report.pdf'}", flush=True)


if __name__ == "__main__":
    main()
