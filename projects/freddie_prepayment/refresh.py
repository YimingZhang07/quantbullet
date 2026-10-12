"""Refresh the prepayment data chain by running the generic procs and turnover stages in order."""

import argparse
from pathlib import Path
import subprocess
import sys

from procs.freddie_sflld.build_sample_panel import read_config as read_sample_config
from procs.freddie_sflld.prepare_panel import read_config as read_panel_config

from .config import REPOSITORY, load_repo_env, read_config


HERE = Path(__file__).resolve().parent
STAGES = ("macro", "convert", "sample", "panel", "frame", "fit", "report")


def plan(sample_config: Path, panel_config: Path, turnover_config: Path) -> dict[str, list[list[str]]]:
    """Map each stage to its ``python -m`` arguments; every data root comes from the three configs."""
    sample = read_sample_config(sample_config)
    panel = read_panel_config(panel_config)
    turnover = read_config(turnover_config)
    if panel.sample_root != sample.output_root:
        raise ValueError(f"Panel sample_root {panel.sample_root} is not the sample output_root {sample.output_root}")
    if turnover.panel_path != panel.output_root / "panel.parquet":
        raise ValueError(f"Turnover panel_path {turnover.panel_path} is not the prepared panel under {panel.output_root}")
    macro = ["--data-root", str(panel.macro_root)]
    return {
        "macro": [["procs.housing_macro.download", *macro, "--refresh"], ["procs.housing_macro.build_parquet", *macro]],
        "convert": [["procs.freddie_sflld.build_parquet", "--data-root", str(sample.freddie_root),
                     "--start", sample.start_vintage, "--end", sample.end_vintage]],
        "sample": [["procs.freddie_sflld.build_sample_panel", "--config", str(sample_config)]],
        "panel": [["procs.freddie_sflld.prepare_panel", "--config", str(panel_config)]],
        "frame": [["projects.freddie_prepayment.prepare_turnover", "--config", str(turnover_config)]],
        "fit": [["projects.freddie_prepayment.fit_turnover", "--config", str(turnover_config)]],
        "report": [["projects.freddie_prepayment.report_turnover", "--config", str(turnover_config)]],
    }


def run(stages: dict[str, list[list[str]]], first: str, last: str, *, dry_run: bool = False) -> int:
    """Run stages ``first`` through ``last`` in separate processes and stop at the first failure."""
    for stage in STAGES[STAGES.index(first):STAGES.index(last) + 1]:
        for command in stages[stage]:
            print(f"[refresh] {stage}: python -m {' '.join(command)}", flush=True)
            if dry_run:
                continue
            code = subprocess.run([sys.executable, "-m", *command], cwd=REPOSITORY).returncode
            if code:
                print(f"[refresh] {stage} FAILED with exit code {code}; later stages skipped", file=sys.stderr, flush=True)
                return code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="first", choices=STAGES, required=True, help="First stage to run")
    parser.add_argument("--to", dest="last", choices=STAGES, default=STAGES[-1], help="Last stage to run (default: report)")
    parser.add_argument("--sample-config", type=Path, default=HERE / "sample_panel.example.toml")
    parser.add_argument("--panel-config", type=Path, default=HERE / "prepare_panel.example.toml")
    parser.add_argument("--turnover-config", type=Path, default=HERE / "turnover.example.toml")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")
    args = parser.parse_args(argv)
    if STAGES.index(args.first) > STAGES.index(args.last):
        parser.error("--from must not come after --to")
    load_repo_env()
    try:
        stages = plan(*(path.expanduser().resolve() for path in (args.sample_config, args.panel_config, args.turnover_config)))
    except Exception as exc:
        print(f"[refresh] FAILED: {exc}", file=sys.stderr)
        return 1
    return run(stages, args.first, args.last, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
