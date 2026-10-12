"""Turnover: LPM (BCD) vs a log-link Poisson GLM on the same cohort, basis and weights.

The GLM reuses the LPM blocks built by ``fit_turnover``: FlatRamp blocks without their bias column
(one global intercept), categorical blocks without their largest-weight level, and the age
interaction as ramps x purpose one-hot. Columns are standardised, and the weighted Poisson MLE is
solved by Newton/IRLS over row chunks, so the dense design is never held in memory.

Run from the repository root:
    .venv/Scripts/python.exe explorations/2026-10-12-lpm-vs-glm/compare_turnover.py \
        --config projects/freddie_prepayment/turnover.example.toml

Outputs go outside the repository, to <output_root>/lpm_vs_glm/<run>/ unless --output-dir is set.
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.backends.backend_pdf import PdfPages
from scipy.linalg import cho_factor, cho_solve
from scipy.special import xlogy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root, for `projects`

from quantbullet.linear_product_model import LinearProductRegressorBCD  # noqa: E402
from projects.freddie_prepayment.config import REPOSITORY, read_config  # noqa: E402
from projects.freddie_prepayment.fit_turnover import (  # noqa: E402
    CATEGORICAL, CLIP, INTERACTIONS, TARGET, build_toolkit, make_container, prediction_metrics, to_model_data,
)
from projects.freddie_prepayment.report_turnover import IMPLIED_BIN_CONFIG, MIN_COUNT  # noqa: E402

SKLEARN_MAX_ROWS = 3_000_000


# ---------------------------------------------------------------------------
# Log-link design and Newton solver
# ---------------------------------------------------------------------------

@dataclass
class Part:
    group: str
    kind: str                      # "numeric" | "categorical" | "interaction"
    by: str | None = None
    keep: np.ndarray = None        # kept columns of the expanded part
    mean: np.ndarray = None
    std: np.ndarray = None
    beta: slice = None             # position in the coefficient vector


class LogLinkDesign:
    """Standardised log-link design built chunk by chunk from LPM blocks; column 0 is the intercept."""

    def __init__(self, feature_groups, blocks, weights, chunk_rows):
        self.chunk_rows = chunk_rows
        self.cached = None
        self.parts = []
        references = {}
        for group, columns in feature_groups.items():
            if group in CATEGORICAL:
                references[group] = int(np.argmax(weights @ blocks[group]))
                self.parts.append(Part(group, "categorical"))
            elif not columns[0].endswith("_bias"):
                raise ValueError(f"{group}: expected a leading FlatRamp bias column")
            elif group in INTERACTIONS:
                self.parts.append(Part(group, "interaction", by=INTERACTIONS[group]))
            else:
                self.parts.append(Part(group, "numeric"))
        self._standardise(blocks, len(weights), references)

    def chunks(self, n):
        for start in range(0, n, self.chunk_rows):
            yield slice(start, min(start + self.chunk_rows, n))

    @staticmethod
    def _expand(part, blocks, rows):
        X = blocks[part.group][rows]
        if part.kind == "categorical":
            return X
        if part.kind == "numeric":
            return X[:, 1:]
        by = blocks[part.by][rows]
        return (X[:, 1:, None] * by[:, None, :]).reshape(len(X), -1)   # ramp-major, level-minor

    def _standardise(self, blocks, n, references):
        sums, squares = {}, {}
        for rows in self.chunks(n):
            for part in self.parts:
                raw = self._expand(part, blocks, rows).astype(np.float64)
                sums[part.group] = sums.get(part.group, 0.) + raw.sum(axis=0)
                squares[part.group] = squares.get(part.group, 0.) + np.einsum("ij,ij->j", raw, raw)
        start = 1
        for part in self.parts:
            mean = sums[part.group] / n
            std = np.sqrt(np.maximum(squares[part.group] / n - mean ** 2, 0.))
            keep = std > 1e-12 * np.maximum(1., np.abs(mean))     # absent levels / constant pieces
            if part.kind == "categorical":
                keep[references[part.group]] = False
            part.keep = np.flatnonzero(keep)
            part.mean, part.std = mean[part.keep], std[part.keep]
            part.beta = slice(start, start + len(part.keep))
            start += len(part.keep)
        self.n_columns = start

    def part_matrix(self, part, blocks, rows):
        return (self._expand(part, blocks, rows)[:, part.keep] - part.mean) / part.std

    def matrix(self, blocks, rows, dtype=np.float64):
        Z = np.empty((rows.stop - rows.start, self.n_columns), dtype=dtype)
        Z[:, 0] = 1.
        for part in self.parts:
            Z[:, part.beta] = self.part_matrix(part, blocks, rows)
        return Z

    def cache(self, blocks, n):
        """Hold the fitted rows' design in float32 so Newton passes skip rebuilding it."""
        self.cached = np.empty((n, self.n_columns), dtype=np.float32)
        for rows in self.chunks(n):
            self.cached[rows] = self.matrix(blocks, rows, dtype=np.float32)

    def fit_rows(self, blocks, rows):
        return self.cached[rows] if self.cached is not None else self.matrix(blocks, rows)

    def part_eta(self, part, blocks, rows, beta):
        return self.part_matrix(part, blocks, rows) @ beta[part.beta]

    def predict(self, blocks, n, beta, fitted_rows=False):
        out = np.empty(n)
        for rows in self.chunks(n):
            Z = self.fit_rows(blocks, rows) if fitted_rows else self.matrix(blocks, rows)
            out[rows] = np.exp(Z @ beta)
        return out


def fit_poisson_newton(design, blocks, y, w, max_iter, tol, l2=0.):
    """Weighted Poisson log-link MLE: Newton/IRLS with step halving; sums accumulate in float64.

    Minimises D(beta) + l2 * |beta without intercept|^2, where D is the balance-weighted mean Poisson
    deviance used by ``prediction_metrics``. A small ``l2`` keeps levels without events finite.
    """
    n, p = len(y), design.n_columns
    beta = np.zeros(p)
    beta[0] = np.log(np.average(y, weights=w))
    w_total = w.sum()
    penalty = np.full(p, l2 * w_total)                # l2 in the scale of the summed log-likelihood
    penalty[0] = 0.
    history, betas, best, step, t = [], [], None, None, 1.
    newton_steps = passes = 0
    status = "max_iter"
    while True:
        deviance, gradient, hessian = 0., np.zeros(p), np.zeros((p, p))
        for rows in design.chunks(n):
            Z = design.fit_rows(blocks, rows)
            eta = Z @ beta
            mu = np.exp(eta)
            yc, wc = y[rows], w[rows]
            deviance += np.sum(wc * 2 * (xlogy(yc, yc) - yc * eta - yc + mu))
            gradient += (wc * (yc - mu)) @ Z                  # float64: the optimum is set by the gradient
            Zs = Z * np.sqrt(wc * mu).astype(Z.dtype)[:, None]
            hessian += Zs.T @ Zs                              # float32 when cached: only steers the step
        deviance /= w_total
        objective = deviance + l2 * (beta[1:] @ beta[1:])
        passes += 1
        if best is not None and objective > best[2]:
            if t < 2 ** -10:
                status = "line_search"
                break
            t /= 2                                    # overshoot: halve the last Newton step
            beta = best[0] + t * step
            continue
        history.append(deviance)
        betas.append(beta.copy())
        previous = best
        best = (beta.copy(), deviance, objective)
        if previous is not None and previous[2] - objective <= tol * previous[2]:
            status = "converged"
            break
        if newton_steps == max_iter:
            break
        system = hessian + np.diag(penalty)
        system[np.diag_indices(p)] += 1e-10 * np.trace(hessian) / p
        rhs = gradient - penalty * beta
        try:
            step = cho_solve(cho_factor(system), rhs)
        except np.linalg.LinAlgError:                 # singular: e.g. sampled levels without events and l2=0
            step = np.linalg.lstsq(system, rhs, rcond=None)[0]
        t = 1.
        beta = best[0] + step
        newton_steps += 1
    return {"beta": best[0], "deviance": best[1], "history": history, "betas": betas, "newton_steps": newton_steps,
            "passes": passes, "status": status}


# ---------------------------------------------------------------------------
# Data, metrics and comparisons
# ---------------------------------------------------------------------------

@dataclass
class Split:
    container: object
    y: np.ndarray
    w: np.ndarray
    month: np.ndarray
    blocks: dict
    preds: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.y)


def floor_deviance(y, pred, w, floor=1e-10):
    """Balance-weighted Poisson deviance with predictions floored as in the LPM solver; defined for negatives."""
    pred = np.maximum(pred, floor)
    return float(np.average(2 * (xlogy(y, y) - xlogy(y, pred) - y + pred), weights=w))


def facet_values(split):
    """Row-aligned group keys; categorical and period keys avoid one Python string per row."""
    data = split.container.orig
    month = pd.Series(split.month).astype("datetime64[ns]")
    facets = {name: data[name] for name in CATEGORICAL}
    facets["d_reporting_month"] = month.dt.to_period("M")
    facets["reporting_year"] = month.dt.year
    for name, width in IMPLIED_BIN_CONFIG.items():
        x = data[name].to_numpy(dtype=float)
        facets[name] = pd.Series(np.round(x) if width == "discrete" else np.floor(x / width + 1e-9) * width)
    return facets


def actual_vs_expected(split_name, split):
    base = pd.DataFrame({"weight": split.w, "actual": split.w * split.y,
                         **{f"pred_{model}": split.w * pred for model, pred in split.preds.items()}})
    base["rows"] = 1
    tables, summary = [], []
    for facet, values in facet_values(split).items():
        table = base.groupby(values, sort=True, observed=True).sum()
        table.index = table.index.astype(str)
        table.index.name = "bin"
        table = table.reset_index()
        for model in split.preds:
            table[f"ae_{model}"] = table["actual"] / table[f"pred_{model}"]
        table.insert(0, "facet", facet)
        table.insert(0, "split", split_name)
        tables.append(table)
        supported = table[table["rows"] >= MIN_COUNT]
        share = supported["weight"] / supported["weight"].sum()
        row = {"split": split_name, "facet": facet, "bins": len(table), "supported_bins": len(supported)}
        for model in split.preds:
            error = supported[f"ae_{model}"] - 1
            row[f"rms_ae_error_{model}"] = float(np.sqrt(np.sum(share * error ** 2)))
            row[f"max_abs_ae_error_{model}"] = float(error.abs().max())
        summary.append(row)
    return pd.concat(tables, ignore_index=True), pd.DataFrame(summary)


def factor_comparison(model, design, beta, toolkit, split):
    """Per-group factors from both models, each normalised to a balance-weighted mean of 1.

    Interaction curves are normalised within each level of the ``by`` group; that level's mean curve
    is folded into the ``by`` categorical factor so both models split level and shape the same way.
    """
    groups = toolkit.feature_groups
    parts = {part.group: part for part in design.parts}
    levels = {name: list(toolkit.preprocess_config[name].categories_[0]) for name in CATEGORICAL}

    # balance-weighted mean of each numeric component (per by-level for interactions)
    glm_mean, lpm_mean = {}, {}
    for group, part in parts.items():
        if part.kind == "categorical":
            continue
        codes = (np.argmax(split.blocks[part.by], axis=1) if part.by else np.zeros(split.n, dtype=int))
        n_levels = len(levels[part.by]) if part.by else 1
        glm_sum = np.zeros(n_levels)
        for rows in design.chunks(split.n):
            glm_sum += np.bincount(codes[rows], weights=split.w[rows] * np.exp(design.part_eta(part, split.blocks, rows, beta)),
                                   minlength=n_levels)
        cat_series = split.container.orig[part.by] if part.by else None
        lpm_component = model._predict_group(group, split.blocks[group], cat_series=cat_series)
        lpm_sum = np.bincount(codes, weights=split.w * lpm_component, minlength=n_levels)
        level_weight = np.bincount(codes, weights=split.w, minlength=n_levels)
        with np.errstate(invalid="ignore", divide="ignore"):
            glm_mean[group], lpm_mean[group] = glm_sum / level_weight, lpm_sum / level_weight

    curves = []
    for group, part in parts.items():
        if part.kind == "categorical":
            continue
        lower, upper = CLIP[group.removesuffix("_fit")]
        grid = np.arange(lower, upper + 1.) if group == "c_age_fit" else np.linspace(lower, upper, 241)
        X = np.asarray(toolkit.preprocess_config[group].transform(pd.DataFrame({group: grid})), dtype=np.float32)
        for index, level in enumerate(levels[part.by] if part.by else [None]):
            if not np.isfinite(glm_mean[group][index]):
                continue                                  # level absent from the fitted rows
            grid_blocks = {group: X}
            if part.by:
                grid_blocks[part.by] = np.tile(np.eye(len(levels[part.by]), dtype=np.float32)[index], (len(grid), 1))
                lpm = X @ model.coef_[group].categories[level]
            else:
                lpm = model._predict_group(group, X)
            glm = np.exp(design.part_eta(part, grid_blocks, slice(0, len(grid)), beta))
            curves.append(pd.DataFrame({"feature": group, "level": level, "x": grid,
                                        "lpm": lpm / lpm_mean[group][index], "glm": glm / glm_mean[group][index]}))
    curves = pd.concat(curves, ignore_index=True)

    factors = []
    for name in CATEGORICAL:
        part, n_levels = parts[name], len(levels[name])
        eye = np.eye(n_levels, dtype=np.float32)
        glm = np.exp(design.part_eta(part, {name: eye}, slice(0, n_levels), beta))
        lpm = eye @ model.coef_[name]
        for group, other in parts.items():
            if other.by == name:
                glm, lpm = glm * glm_mean[group], lpm * lpm_mean[group]
        weight = split.w @ split.blocks[name]
        share = weight / weight.sum()
        seen = share > 0                                  # absent levels have no fitted factor
        lpm, glm = np.where(seen, lpm, np.nan), np.where(seen, glm, np.nan)
        factors.append(pd.DataFrame({"feature": name, "level": levels[name], "weight_share": share,
                                     "lpm": lpm / (share[seen] @ lpm[seen]), "glm": glm / (share[seen] @ glm[seen])}))
    factors = pd.concat(factors, ignore_index=True)
    for table in (curves, factors):
        table["glm_over_lpm"] = table["glm"] / table["lpm"]
    return curves, factors


def plot_factors(curves, factors, split, path):
    with PdfPages(path) as pdf:
        features = list(dict.fromkeys(curves["feature"]))
        n_cols = 3
        fig, axes = plt.subplots(-(-len(features) // n_cols), n_cols, figsize=(14, 4 * -(-len(features) // n_cols)))
        for ax, feature in zip(axes.flat, features):
            x_data = split.container.orig[feature].to_numpy()
            low, high = np.quantile(x_data, [.01, .99])
            for i, (level, curve) in enumerate(curves[curves["feature"] == feature].groupby("level", dropna=False, sort=False)):
                label = "" if pd.isna(level) else f" {level}"
                ax.plot(curve["x"], curve["lpm"], color=f"C{i}", label=f"LPM{label}")
                ax.plot(curve["x"], curve["glm"], color=f"C{i}", linestyle="--", label=f"GLM{label}")
            ax.axvspan(curve["x"].min(), low, color="0.9", zorder=0)
            ax.axvspan(high, curve["x"].max(), color="0.9", zorder=0)
            ax.set_title(f"{feature} (gray: outside 1-99% of rows)", fontsize=9)
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(features):]:
            ax.set_visible(False)
        fig.suptitle("Numeric factors, balance-weighted mean 1 (solid LPM, dashed GLM)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        layout = [["f_purpose", "f_occupancy", "f_property_type"], ["f_first_time_buyer", "f_month", "f_month"],
                  ["f_state"] * 3]
        fig, axes = plt.subplot_mosaic(layout, figsize=(14, 11))
        for name, ax in axes.items():
            table = factors[factors["feature"] == name]
            x = np.arange(len(table))
            ax.plot(x, table["lpm"], "o", label="LPM")
            ax.plot(x, table["glm"], "x", label="GLM")
            ax.set_xticks(x, table["level"], rotation=90 if len(table) > 12 else 0, fontsize=7)
            ax.set_title(name, fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle("Categorical factors, balance-weighted mean 1")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", help="Default: <output_root>/lpm_vs_glm/<run>")
    parser.add_argument("--sample-rows", type=int, help="Random loan-month subset, as fit_turnover --smoke-rows")
    parser.add_argument("--holdout-months", type=int, default=0,
                        help="Fit without the last N reporting months and score them out of time")
    parser.add_argument("--lpm-ftol", type=float, help="Override the config ftol for the LPM fit")
    parser.add_argument("--sklearn-check", action="store_true",
                        help=f"Refit the GLM with sklearn newton-cholesky (at most {SKLEARN_MAX_ROWS:,} training rows)")
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument("--max-newton-iter", type=int, default=50)
    parser.add_argument("--newton-tol", type=float, default=1e-10,
                        help="Relative objective decrease below which Newton stops")
    parser.add_argument("--glm-l2", type=float, default=0.,
                        help="Ridge on standardised GLM coefficients; ~1e-8 keeps sampled levels without events finite")
    parser.add_argument("--stream-design", action="store_true",
                        help="Rebuild the GLM design every pass instead of caching it in float32 (rows x ~156 x 4 bytes)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_config(args.config)
    lpm_ftol = args.lpm_ftol if args.lpm_ftol is not None else config.ftol
    run = ("full" if args.sample_rows is None else f"sample{args.sample_rows}") + \
          ("_insample" if not args.holdout_months else f"_holdout{args.holdout_months}m")
    if args.lpm_ftol is not None:
        run += f"_ftol{args.lpm_ftol:g}"
    if args.glm_l2:
        run += f"_l2{args.glm_l2:g}"
    out = Path(args.output_dir).resolve() if args.output_dir else config.output_root / "lpm_vs_glm" / run
    if out.is_relative_to(REPOSITORY):
        raise ValueError("Write comparison outputs outside the repository")
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    started = time.perf_counter()
    frame = pl.read_parquet(config.output_root / "turnover_frame.parquet")
    if args.sample_rows is not None:
        frame = frame.sample(n=min(args.sample_rows, frame.height), seed=42, shuffle=True)
    month = frame["d_reporting_month"].to_numpy()
    data = to_model_data(frame)
    del frame
    toolkit = build_toolkit(data)
    groups = toolkit.feature_groups
    masks = {"train": np.ones(len(data), dtype=bool)}
    cutoff = None
    if args.holdout_months:
        unique = np.unique(month)
        if not 0 < args.holdout_months < len(unique):
            raise ValueError(f"--holdout-months must be between 1 and {len(unique) - 1}")
        cutoff = unique[-args.holdout_months]
        masks = {"train": month < cutoff, "test": month >= cutoff}
    splits = {}
    for name, mask in masks.items():
        part = data.loc[mask].reset_index(drop=True)
        container = make_container(part, toolkit)
        splits[name] = Split(container, part[TARGET].to_numpy(dtype=float), part["weight"].to_numpy(dtype=float),
                             month[mask], container.get_expanded_array_dict(list(groups)))
    del data
    train = splits["train"]
    timings["data"] = time.perf_counter() - started
    print(f"[compare] rows: " + ", ".join(f"{k}={v.n:,}" for k, v in splits.items())
          + (f"; test months >= {pd.Timestamp(cutoff):%Y-%m}" if cutoff is not None else ""), flush=True)

    # ---- LPM, exactly as fit_turnover (optionally with another ftol)
    model = LinearProductRegressorBCD()
    started = time.perf_counter()
    model.fit(train.container, feature_groups=groups, interactions=INTERACTIONS, init_params=None,
              n_iterations=config.n_iterations, early_stopping_rounds=config.early_stopping_rounds, ftol=lpm_ftol,
              cache_qr_decomp=False, loss="poisson", weights=train.w)
    timings["lpm_fit"] = time.perf_counter() - started
    for split in splits.values():
        split.preds["lpm"] = model.predict(split.container)

    # ---- log-link GLM on the same blocks
    started = time.perf_counter()
    design = LogLinkDesign(groups, train.blocks, train.w, args.chunk_rows)
    if not args.stream_design:
        design.cache(train.blocks, train.n)
    timings["glm_design"] = time.perf_counter() - started
    print(f"[compare] GLM design: {design.n_columns} columns, "
          f"{'cached float32' if design.cached is not None else 'streamed'}; built in {timings['glm_design']:.1f}s", flush=True)
    no_events = [f"{name}={level}" for name in CATEGORICAL
                 for level, weight, events in zip(toolkit.preprocess_config[name].categories_[0],
                                                  train.w @ train.blocks[name], train.y @ train.blocks[name])
                 if weight > 0 and events == 0]
    if no_events and not args.glm_l2:
        print(f"[compare] levels without events {no_events}: their MLE is -inf, Newton will crawl; "
              f"consider --glm-l2 1e-8", flush=True)
    started = time.perf_counter()
    glm = fit_poisson_newton(design, train.blocks, train.y, train.w, args.max_newton_iter, args.newton_tol, args.glm_l2)
    timings["glm_newton"] = time.perf_counter() - started
    print(f"[compare] GLM Newton: {glm['newton_steps']} steps, {glm['passes']} passes, "
          f"status={glm['status']}, {timings['glm_newton']:.1f}s", flush=True)
    for name, split in splits.items():
        split.preds["glm"] = design.predict(split.blocks, split.n, glm["beta"], fitted_rows=name == "train")
    design.cached = None

    sklearn_check = None
    if args.sklearn_check:
        if train.n > SKLEARN_MAX_ROWS:
            raise ValueError(f"--sklearn-check needs at most {SKLEARN_MAX_ROWS:,} training rows; use --sample-rows")
        from sklearn.linear_model import PoissonRegressor
        Z = np.vstack([design.matrix(train.blocks, rows) for rows in design.chunks(train.n)])
        started = time.perf_counter()
        # sklearn minimises D/2 + alpha/2 |coef|^2 with an unpenalised intercept: the same optimum as alpha = l2.
        reference = PoissonRegressor(alpha=args.glm_l2, fit_intercept=True, solver="newton-cholesky", max_iter=200,
                                     tol=1e-10).fit(Z[:, 1:], train.y, sample_weight=train.w)
        seconds = time.perf_counter() - started
        eta = reference.intercept_ + Z[:, 1:] @ reference.coef_
        deviance = np.average(2 * (xlogy(train.y, train.y) - train.y * eta - train.y + np.exp(eta)), weights=train.w)
        sklearn_check = {"seconds": seconds, "iterations": int(reference.n_iter_), "deviance": float(deviance),
                         "deviance_minus_newton": float(deviance - glm["deviance"]),
                         "max_abs_eta_diff": float(np.max(np.abs(eta - Z @ glm["beta"])))}
        del Z

    # ---- comparisons
    started = time.perf_counter()
    metrics = {}
    for split_name, split in splits.items():
        for model_name, pred in split.preds.items():
            metrics[f"{split_name}/{model_name}"] = prediction_metrics(split.y, pred, split.w)
            metrics[f"{split_name}/{model_name}"]["floor_poisson_deviance"] = floor_deviance(split.y, pred, split.w)
    tables = [actual_vs_expected(name, split) for name, split in splits.items()]
    ae_bins = pd.concat([t[0] for t in tables], ignore_index=True)
    ae_summary = pd.concat([t[1] for t in tables], ignore_index=True)
    curves, factors = factor_comparison(model, design, glm["beta"], toolkit, train)
    timings["comparisons"] = time.perf_counter() - started

    summary = {
        "rows": {name: split.n for name, split in splits.items()},
        "sample_rows": args.sample_rows, "holdout_months": args.holdout_months,
        "test_from_month": None if cutoff is None else f"{pd.Timestamp(cutoff):%Y-%m}",
        "lpm": {"n_iterations": config.n_iterations, "early_stopping_rounds": config.early_stopping_rounds,
                "ftol": lpm_ftol, "sweeps": len(model.loss_history_), "best_iteration": int(model.best_iteration_) + 1,
                "convergence": model.convergence_info_, "loss_history": model.loss_history_},
        "glm": {"columns": design.n_columns, "l2": args.glm_l2, "newton_steps": glm["newton_steps"],
                "passes": glm["passes"], "status": glm["status"], "deviance_history": glm["history"],
                "levels_without_events": no_events,
                "chunk_rows": args.chunk_rows, "cached_design": not args.stream_design},
        "sklearn_check": sklearn_check, "timings_seconds": timings, "metrics": metrics,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    ae_bins.to_csv(out / "ae_bins.csv", index=False)
    ae_summary.to_csv(out / "ae_summary.csv", index=False)
    curves.to_csv(out / "factor_curves.csv", index=False)
    factors.to_csv(out / "categorical_factors.csv", index=False)
    plot_factors(curves, factors, train, out / "factors.pdf")

    print("\n[compare] model  split   fit_s  sweeps/steps  deviance(floored)  deviance(strict)     A/E  negative")
    for split_name in splits:
        for model_name, seconds, steps in (("lpm", timings["lpm_fit"], len(model.loss_history_)),
                                           ("glm", timings["glm_design"] + timings["glm_newton"], glm["newton_steps"])):
            m = metrics[f"{split_name}/{model_name}"]
            weighted = m.get("balance_weighted", {})
            strict = weighted.get("poisson_deviance")
            print(f"          {model_name:5s}  {split_name:5s}  {seconds:6.1f}  {steps:12d}  "
                  f"{m['floor_poisson_deviance']:17.8f}  {float('nan') if strict is None else strict:16.8f}  "
                  f"{weighted.get('actual_over_expected', float('nan')):6.4f}  {m['prediction_range']['negative']:>8d}")
    if sklearn_check:
        print(f"[compare] sklearn check: {sklearn_check}")
    columns = ["split", "facet", "supported_bins", "rms_ae_error_lpm", "rms_ae_error_glm",
               "max_abs_ae_error_lpm", "max_abs_ae_error_glm"]
    print("\n[compare] A/E error by facet (bins with >= {} rows, balance-weighted)".format(MIN_COUNT))
    print(ae_summary[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    worst = factors[factors["weight_share"] >= 1e-3]
    worst = worst.assign(gap=(worst["glm_over_lpm"] - 1).abs()).sort_values("gap", ascending=False)
    print("\n[compare] largest categorical factor gaps (GLM / LPM), levels with >= 0.1% of balance")
    print(worst.head(8)[["feature", "level", "weight_share", "lpm", "glm", "glm_over_lpm"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[compare] outputs written to {out}")


if __name__ == "__main__":
    main()
