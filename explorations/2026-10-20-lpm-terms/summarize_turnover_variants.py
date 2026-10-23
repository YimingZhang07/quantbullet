"""Summarize turnover variants written by report_turnover_variant.py: one Markdown table of fits, one of predictions.

Each variant is a folder under ROOT (``{basis}_{reference|mean1}_{link}``) holding turnover_model.pkl and
turnover_predictions.parquet. Prediction differences compare variants of the same link: tent with references and
tent at mean 1 against FlatRamp, row by row on the saved keys.

Run from the repository root:
    .venv/Scripts/python.exe explorations/2026-10-20-lpm-terms/summarize_turnover_variants.py ROOT
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

VARIANTS = ("tent_reference", "tent_mean1", "flat_ramp")
KEYS = ["row_id", "loan_identifier", "d_reporting_month"]


def load(root, name):
    with (root / name / "turnover_model.pkl").open("rb") as file:
        bundle = pickle.load(file)
    return bundle["model"], bundle["meta"], pl.read_parquet(root / name / "turnover_predictions.parquet")


def main():
    root = Path(sys.argv[1])
    print("| variant | form | iterations | fit s | deviance (balance) | A/E balance | A/E equal | global scalar |")
    print("|---|---|---|---|---|---|---|---|")
    predictions = {}
    for link in ("log", "identity"):
        for variant in VARIANTS:
            name = f"{variant}_{link}"
            model, meta, pred = load(root, name)
            metrics = meta["metrics"]["balance_weighted"]
            print(f"| {name} | {meta['form']} | {meta['actual_sweeps']} | {meta['fit_seconds']:.0f} | "
                  f"{metrics['poisson_deviance']:.10f} | {metrics['actual_over_expected']:.6f} | "
                  f"{meta['metrics']['equal_weight']['actual_over_expected']:.6f} | {meta['global_scalar']:.6f} |")
            predictions[name] = pred
    print()
    print("| link | compared | max relative difference | p99 | median |")
    print("|---|---|---|---|---|")
    for link in ("log", "identity"):
        flat = predictions[f"flat_ramp_{link}"]
        for variant in VARIANTS[:2]:
            joined = predictions[f"{variant}_{link}"].join(flat, on=KEYS, how="inner", validate="1:1", suffix="_flat")
            relative = np.abs(joined["pred_turnover"].to_numpy() / joined["pred_turnover_flat"].to_numpy() - 1)
            print(f"| {link} | {variant} vs flat_ramp | {relative.max():.2e} | {np.quantile(relative, .99):.2e} | "
                  f"{np.median(relative):.2e} |")


if __name__ == "__main__":
    main()
