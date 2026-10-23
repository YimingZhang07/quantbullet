"""Turnover with tent terms and references against the FlatRamp turnover: same cohort, same fit settings.

On the clipped inputs a tent with knots (clip lower, *KNOTS, clip upper) spans the same curves as FlatRamp with
a bias, so the two fits should agree; the tent model reads as multipliers at the knots, 1 at each reference.
Reads the prepared frame of the config's output_root and writes only to --out, outside the repository:
summary.json and knot_tables.csv.

Run from the repository root (one line):
    .venv/Scripts/python.exe explorations/2026-10-20-lpm-terms/compare_tent_turnover.py
        --config projects/freddie_prepayment/turnover.example.toml --out <directory> [--smoke-rows N --l2 1e-8]

A small sample has levels without events, whose likelihood is maximized at a factor of 0: Newton then chases
infinite coefficients and the two fits stop at different points. --l2 (Newton's ridge) keeps them finite.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from projects.freddie_prepayment.config import REPOSITORY, read_config  # noqa: E402
from projects.freddie_prepayment.fit_turnover import (CATEGORICAL, CLIP, KNOTS, TARGET,  # noqa: E402
                                                      prediction_metrics, solver_arguments, to_model_data)
from quantbullet.linear_product_model import LinearProductRegressor, ModelSpec  # noqa: E402
from quantbullet.linear_product_model.terms import BasisTerm, ByTerm, CategoricalTerm, TentTerm  # noqa: E402
from quantbullet.preprocessing.transformers import FlatRampTransformer  # noqa: E402

# A 24-month-old loan 1pp out of the money, at zero SATO, FICO 740, original and updated LTV 80, $300k (2025
# dollars), 90% of its original balance and unchanged home prices. Every reference is a knot.
REFERENCES = {"c_age": 24, "c_incentive": -1., "c_sato": 0., "c_orig_fico": 740, "c_orig_ltv": 80,
              "c_updated_ltv": 80, "c_orig_balance_real": 300000, "c_factor": .9, "c_hpi_ratio": 1.}


def flat_ramp_spec() -> ModelSpec:
    """The turnover spec before it moved to tents: FlatRamp (with bias) numeric terms on the interior knots."""
    categorical = {name: CategoricalTerm(name) for name in CATEGORICAL}
    numeric = {name + "_fit": BasisTerm(name + "_fit", FlatRampTransformer(knots=knots, include_bias=True))
               for name, knots in KNOTS.items()}
    numeric["c_age_fit"] = ByTerm(numeric["c_age_fit"], by=categorical["f_purpose"], drop=("c_age_fit",))
    return ModelSpec([*categorical.values(), *numeric.values()])


def tent_spec(references=REFERENCES) -> ModelSpec:
    """The turnover spec (``build_spec``: tents whose end knots are the clip bounds) plus ``references``
    (features without one keep the mean-1 scale)."""
    categorical = {name: CategoricalTerm(name) for name in CATEGORICAL}
    numeric = {name + "_fit": TentTerm(name + "_fit", (CLIP[name][0], *KNOTS[name], CLIP[name][1]),
                                       reference=references.get(name)) for name in KNOTS}
    numeric["c_age_fit"] = ByTerm(numeric["c_age_fit"], by=categorical["f_purpose"], drop=("c_age_fit",))
    return ModelSpec([*categorical.values(), *numeric.values()])


def run(spec, data, config, l2=None):
    started = time.perf_counter()
    prepared = spec.prepare(data, response=TARGET, weights="weight", dtype=np.float32)
    prepare_seconds = time.perf_counter() - started
    started = time.perf_counter()
    arguments = solver_arguments(config)
    if l2 is not None:
        arguments["solver_options"] = {**arguments["solver_options"], "l2": l2}
    model = LinearProductRegressor().fit(prepared, loss="poisson", block_link=config.block_link, verbose=0,
                                         **arguments)
    fit_seconds = time.perf_counter() - started
    pred = model.predict(prepared)
    info = {"prepare_seconds": prepare_seconds, "fit_seconds": fit_seconds,
            "design_columns": model.fit_timing_.get("design_columns", prepared.shape[1]),
            "steps": model.convergence_info_["iteration"], "reason": model.convergence_info_["reason"],
            "deviance": model.convergence_info_.get("deviance"), "loss": model.loss_history_[-1],
            "global_scalar": model.global_scalar_, "pivots": model.convergence_info_.get("pivots"),
            "metrics": prediction_metrics(data[TARGET], pred, data["weight"])}
    del prepared
    gc.collect()
    return model, pred, info


def knot_tables(tent, flat):
    """Tent multipliers at the knots against the FlatRamp curve at the same points, both 1 at the reference."""
    rows = []
    for name in KNOTS:
        group = name + "_fit"
        knots = [CLIP[name][0], *KNOTS[name], CLIP[name][1]]
        tent_term, flat_term = tent.term(group), flat.term(group)
        levels = list(tent.coef_[group].categories) if group in tent.interactions_ else [None]
        for level in levels:
            tent_curve = tent_term.curve(knots, level=level)
            flat_curve = flat_term.curve(knots, level=level) / flat_term.curve([REFERENCES[name]], level=level)[0]
            rows += [{"feature": group, "level": level, "knot": knot, "tent": a, "flat_ramp": b}
                     for knot, a, b in zip(knots, tent_curve, flat_curve)]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="Directory for summary.json and knot_tables.csv")
    parser.add_argument("--smoke-rows", type=int)
    parser.add_argument("--l2", type=float, help="Newton ridge for both fits, for samples with levels without events")
    args = parser.parse_args()
    config = read_config(args.config)
    out = Path(args.out).resolve()
    if out.is_relative_to(REPOSITORY):
        raise ValueError("--out must be outside the repository")
    out.mkdir(parents=True, exist_ok=True)

    frame = pl.read_parquet(config.output_root / "turnover_frame.parquet")
    if args.smoke_rows is not None:
        frame = frame.sample(n=min(args.smoke_rows, frame.height), seed=42, shuffle=True)
    data = to_model_data(frame)
    del frame
    print(f"[compare] {len(data):,} rows; {config.block_link} form", flush=True)

    flat, flat_pred, flat_info = run(flat_ramp_spec(), data, config, args.l2)
    print(f"[compare] FlatRamp: {flat_info['steps']} steps, deviance {flat_info['deviance']}, "
          f"{flat_info['fit_seconds']:.0f}s", flush=True)
    tent, tent_pred, tent_info = run(tent_spec(), data, config, args.l2)
    print(f"[compare] tent: {tent_info['steps']} steps, deviance {tent_info['deviance']}, "
          f"{tent_info['fit_seconds']:.0f}s", flush=True)

    positive = (flat_pred > 0) & (tent_pred > 0)
    relative = np.abs(tent_pred[positive] / flat_pred[positive] - 1)
    tables = knot_tables(tent, flat)
    tables.to_csv(out / "knot_tables.csv", index=False)
    summary = {"rows": len(data), "block_link": config.block_link, "smoke_rows": args.smoke_rows, "l2": args.l2,
               "references": REFERENCES, "flat_ramp": flat_info, "tent": tent_info,
               "prediction_relative_difference": {"max": float(relative.max()),
                                                  "p99": float(np.quantile(relative, .99)),
                                                  "median": float(np.median(relative)),
                                                  "rows_compared": int(positive.sum()),
                                                  "rows_zero_in_one_model": int(((flat_pred > 0) != (tent_pred > 0)).sum())},
               "knot_table_max_abs_difference": float(np.abs(tables["tent"] - tables["flat_ramp"]).max())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({key: summary[key] for key in ("prediction_relative_difference",
                                                    "knot_table_max_abs_difference")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
