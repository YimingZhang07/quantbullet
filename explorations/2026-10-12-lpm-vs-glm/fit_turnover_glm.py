"""Fit the turnover cohort with the log-link GLM and save artifacts that report_turnover reads unchanged.

Same data, toolkit and container as fit_turnover; the GLM and its Newton solver come from compare_turnover.
The fitted GLM is expressed exactly as a LinearProductRegressorBCD:

- numeric block k: a frozen submodel, sklearn PoissonRegressor with coef_/intercept_ set so that
  predict(X_k) = exp(X_k @ a_k + b_k) / mean, i.e. the GLM component at balance-weighted mean 1;
- age x purpose: an InteractionCoef holding one such submodel per purpose;
- categorical block: factor vector exp(eta_level) / mean; purpose also carries each purpose's mean age level;
- global_scalar_: intercept and all means, so predict() equals exp(Z beta).

Only standard quantbullet / sklearn classes are pickled. Run from the repository root:
    .venv/Scripts/python.exe explorations/2026-10-12-lpm-vs-glm/fit_turnover_glm.py \
        --config projects/freddie_prepayment/turnover.example.toml
then run report_turnover with the printed config. Artifacts go to <output_root>_glm unless --output-root is set.
"""
import argparse
import os
import pickle
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import PoissonRegressor

from compare_turnover import LogLinkDesign, fit_poisson_newton   # also puts the repository root on sys.path
from quantbullet.linear_product_model import InteractionCoef, LinearProductRegressorBCD
from quantbullet.linear_product_model.base import FitMetadata
from quantbullet.utils.files import file_sha256, temporary_output
from projects.freddie_prepayment.config import REPOSITORY, read_config
from projects.freddie_prepayment.fit_turnover import (
    CATEGORICAL, CLIP, INTERACTIONS, KEYS, KNOTS, MODEL_INPUTS, TARGET, build_toolkit, make_container,
    prediction_metrics, to_model_data,
)


def exp_linear(slope, intercept):
    """A PoissonRegressor whose predict(X) is exp(X @ slope + intercept)."""
    model = PoissonRegressor(alpha=0., max_iter=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(np.zeros((2, len(slope))), np.ones(2))   # sets the fitted state; coefficients replaced below
    model.coef_, model.intercept_ = np.asarray(slope, dtype=float), float(intercept)
    return model


def lpm_components(design, beta, groups, levels, blocks, w, codes):
    """Express a GLM coefficient vector as LPM ``coef_``, ``submodels_`` and ``global_scalar_``."""
    linear = {}
    for part in design.parts:
        n_columns = len(groups[part.group])
        if part.kind != "categorical":
            n_columns = (n_columns - 1) * (len(groups[part.by]) if part.by else 1)   # without the bias column
        slope = np.zeros(n_columns)
        slope[part.keep] = beta[part.beta] / part.std
        linear[part.group] = (slope, -float(beta[part.beta] @ (part.mean / part.std)))

    coef, submodels, level_means = {}, {}, {}
    scalar = float(np.exp(beta[0]))
    for part in design.parts:
        if part.kind == "categorical":
            continue
        slope, intercept = linear[part.group]
        X = blocks[part.group]
        if part.kind == "numeric":
            slope = np.concatenate([[0.], slope])
            mean = np.average(np.exp(X @ slope + intercept), weights=w)
            submodels[part.group] = exp_linear(slope, intercept - np.log(mean))
            coef[part.group] = np.full(len(slope), np.nan)        # placeholder: the submodel predicts this block
            scalar *= mean
            continue
        slopes = slope.reshape(-1, len(groups[part.by]))           # ramp-major, level-minor
        curves, means = {}, np.ones(len(groups[part.by]))
        for index, level in enumerate(levels[part.by]):
            mask = codes[part.by] == index
            if not mask.any():
                continue
            level_slope = np.concatenate([[0.], slopes[:, index]])
            means[index] = np.average(np.exp(X[mask] @ level_slope + intercept), weights=w[mask])
            curves[level] = exp_linear(level_slope, intercept - np.log(means[index]))
        coef[part.group] = InteractionCoef(by=part.by, categories=curves)
        level_means[part.by] = means

    for part in design.parts:
        if part.kind != "categorical":
            continue
        slope, intercept = linear[part.group]
        factor = np.exp(slope + intercept) * level_means.get(part.group, 1.)
        share = np.bincount(codes[part.group], weights=w, minlength=len(factor)) / w.sum()
        mean = float(share @ factor)
        coef[part.group] = factor / mean
        scalar *= mean
    return coef, submodels, scalar


def link_or_copy(source, target):
    if target.exists():
        if os.path.samefile(source, target):
            return
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", help="Default: <output_root>_glm (_glm_smoke<N> with --smoke-rows)")
    parser.add_argument("--smoke-rows", type=int, help="Explicit diagnostic subset, as fit_turnover")
    parser.add_argument("--glm-l2", type=float, default=0., help="See compare_turnover; ~1e-8 for subsets")
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument("--max-newton-iter", type=int, default=50)
    parser.add_argument("--newton-tol", type=float, default=1e-10)
    args = parser.parse_args()

    config = read_config(args.config)
    default = config.output_root.name + ("_glm" if args.smoke_rows is None else f"_glm_smoke{args.smoke_rows}")
    root = Path(args.output_root).resolve() if args.output_root else config.output_root.with_name(default)
    if root.is_relative_to(REPOSITORY) or root == config.output_root:
        raise ValueError("GLM artifacts need their own directory outside the repository")
    root.mkdir(parents=True, exist_ok=True)

    source = config.output_root / "turnover_frame.parquet"
    frame = pl.read_parquet(source)
    if args.smoke_rows is not None:
        frame = frame.sample(n=min(args.smoke_rows, frame.height), seed=42, shuffle=True)
    data = to_model_data(frame)
    toolkit = build_toolkit(data)
    container = make_container(data, toolkit)
    groups = toolkit.feature_groups
    blocks = container.get_expanded_array_dict(list(groups))
    y, w = data[TARGET].to_numpy(dtype=float), data["weight"].to_numpy(dtype=float)
    levels = {name: list(toolkit.preprocess_config[name].categories_[0]) for name in CATEGORICAL}
    codes = {name: np.argmax(blocks[name], axis=1) for name in CATEGORICAL}
    print(f"[glm] {len(y):,} rows; fitting log-link Poisson GLM", flush=True)

    started = time.perf_counter()
    design = LogLinkDesign(groups, blocks, w, args.chunk_rows)
    design.cache(blocks, len(y))
    glm = fit_poisson_newton(design, blocks, y, w, args.max_newton_iter, args.newton_tol, args.glm_l2)
    seconds = time.perf_counter() - started
    design.cached = None
    print(f"[glm] {design.n_columns} columns; {glm['newton_steps']} Newton steps, status={glm['status']}, "
          f"{seconds:.1f}s", flush=True)
    if glm["status"] != "converged":
        warnings.warn(f"Newton stopped with status {glm['status']}")

    coef, submodels, scalar = lpm_components(design, glm["beta"], groups, levels, blocks, w, codes)
    scalars = [lpm_components(design, beta, groups, levels, blocks, w, codes)[2] for beta in glm["betas"]]
    model = LinearProductRegressorBCD()
    model.feature_groups_, model.coef_, model.submodels_ = groups, coef, submodels
    model.interactions_, model.global_scalar_, model.loss_, model.offset_y = dict(INTERACTIONS), scalar, "poisson", None
    model.fit_metadata_ = FitMetadata(n_obs=len(y), n_features=container.shape[1], offset_y=None,
                                      mean_y=float(np.mean(y)), ftol=args.newton_tol)
    model.loss_history_, model.global_scalar_history_ = glm["history"], scalars
    model.global_scalar_step_history_ = [(step, "newton", value) for step, value in enumerate(scalars)]
    model.best_loss_, model.best_iteration_ = glm["deviance"], len(glm["history"]) - 1
    model.convergence_info_ = {"reason": glm["status"], "iteration": glm["newton_steps"],
                               "current_loss": glm["deviance"], "solver": "log-link GLM Newton"}

    pred = model.predict(container)
    direct = design.predict(blocks, len(y), glm["beta"])
    gap = float(np.max(np.abs(pred / direct - 1)))
    if gap > 1e-6:
        raise RuntimeError(f"LPM form differs from the GLM by {gap:.2e}")
    metrics = prediction_metrics(y, pred, w)
    meta = {"rows": frame.height, "expanded_columns": design.n_columns, "fit_seconds": seconds,
            "loss": "poisson / log-link GLM", "model": "log-link GLM (Newton), exploration",
            "weight_source": "c_prev_balance", "target": TARGET,
            "clips": {name: [lower, upper] for name, (lower, upper) in CLIP.items()},
            "clip_counts": {name: int(((frame[name] < lower) | (frame[name] > upper)).sum())
                            for name, (lower, upper) in CLIP.items()},
            "knots": {name: list(knots) for name, knots in KNOTS.items()},
            "categorical": list(CATEGORICAL), "model_inputs": list(MODEL_INPUTS), "interactions": dict(INTERACTIONS),
            "categories": {name: list(map(str, levels[name])) for name in CATEGORICAL},
            "incentive_max": config.incentive_max, "n_iterations": args.max_newton_iter,
            "early_stopping_rounds": None, "ftol": args.newton_tol, "glm_l2": args.glm_l2,
            "smoke_rows": args.smoke_rows, "frame_sha256": file_sha256(source), "global_scalar": scalar,
            "best_loss": float(glm["deviance"]), "best_iteration": glm["newton_steps"], "metrics": metrics,
            "actual_sweeps": glm["passes"], "lpm_form_max_relative_gap": gap}

    link_or_copy(source, root / "turnover_frame.parquet")
    predictions = frame.select(*KEYS).with_columns(pl.Series("pred_turnover", pred))
    with temporary_output(root / "turnover_predictions.parquet") as path:
        predictions.write_parquet(path, compression="zstd")
        meta["predictions_sha256"] = file_sha256(path)
    with temporary_output(root / "turnover_model.pkl") as path:
        with path.open("wb") as file:
            pickle.dump({"model": model, "toolkit": toolkit, "meta": meta}, file)
    report_config = root / "turnover_glm.toml"
    report_config.write_text(
        f"[data]\npanel_path = '{config.panel_path.as_posix()}'\noutput_root = '{root.as_posix()}'\n\n"
        f"[cohort]\nincentive_max = {config.incentive_max}\n")
    print(f"[glm] metrics={metrics}", flush=True)
    print(f"[glm] artifacts saved; LPM form matches the GLM to {gap:.1e}. Report with:\n"
          f"  .venv/Scripts/python.exe -m projects.freddie_prepayment.report_turnover --config {report_config}",
          flush=True)


if __name__ == "__main__":
    main()
