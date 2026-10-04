"""Stage 2: fit the prepared cohort; persist a reusable model and aligned predictions."""

import argparse
import gc
import pickle
import time

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import OneHotEncoder

from quantbullet.linear_product_model import LinearProductModelToolkit, LinearProductRegressorBCD
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
from quantbullet.model.feature import DataType, Feature, FeatureRole, FeatureSpec
from quantbullet.preprocessing.transformers import FlatRampTransformer
from quantbullet.utils.files import file_sha256, temporary_output

from .config import Config, read_config


TARGET = "y_full_prepay"
CATEGORICAL = ("f_purpose", "f_occupancy", "f_property_type", "f_first_time_buyer", "f_month", "f_state")
KEYS = ("row_id", "loan_identifier", "d_reporting_month")
# Model inputs only. The prepared frame keeps the raw columns.
CLIP = {
    "c_age": (1, 120),
    "c_incentive": (-5, -.5),
    "c_orig_fico": (620, 800),
    "c_updated_ltv": (5, 100),
    "c_orig_balance": (25000, 800_000),
    "c_prev_balance": (25000, 800_000),
    "c_hpi_ratio": (.8, 2.0),
}
KNOTS = {
    "c_age": (3, 6, 12, 18, 24, 36, 60, 84, 108),
    "c_incentive": (-4, -3, -2, -1.5, -1, -.75),
    "c_orig_fico": (660, 700, 740, 780),
    "c_updated_ltv": (20, 40, 60, 80, 95),
    "c_orig_balance": (100000, 150000, 250000, 350000, 500000, 700000),
    "c_prev_balance": (100000, 150000, 250000, 350000, 500000, 700000),
    "c_hpi_ratio": (.95, 1, 1.1, 1.25, 1.5, 1.75),
}
FIT_NUMERIC = tuple(name + "_fit" for name in KNOTS)
MODEL_INPUTS = (*FIT_NUMERIC, *CATEGORICAL)
INTERACTIONS = {"c_age_fit": "f_purpose"}


def to_model_data(frame: pl.DataFrame) -> pd.DataFrame:
    """Clip numeric inputs here so a clip change does not rewrite the prepared frame."""
    if tuple(CLIP) != tuple(KNOTS):
        raise ValueError("CLIP and KNOTS must name the same features in the same order")
    data = {}
    for name, (lower, upper) in CLIP.items():
        data[name + "_fit"] = frame[name].clip(lower, upper).cast(pl.Float32).to_numpy()
    for name in (*CATEGORICAL, TARGET, "weight"):
        values = frame[name].to_numpy()
        data[name] = pd.Categorical(values) if name in CATEGORICAL else values
    return pd.DataFrame(data).reset_index(drop=True)


def build_toolkit(data: pd.DataFrame) -> LinearProductModelToolkit:
    transformers = {name: OneHotEncoder(drop=None, handle_unknown="error", dtype=np.float32) for name in CATEGORICAL}
    transformers.update({name + "_fit": FlatRampTransformer(knots=knots, include_bias=True) for name, knots in KNOTS.items()})
    features = [Feature(name, DataType.CATEGORY, FeatureRole.MODEL_INPUT) for name in CATEGORICAL]
    features += [Feature(name + "_fit", DataType.FLOAT, FeatureRole.MODEL_INPUT) for name in KNOTS]
    features.append(Feature(TARGET, DataType.FLOAT, FeatureRole.TARGET))
    return LinearProductModelToolkit(FeatureSpec(features), preprocess_config=transformers).fit(data)


def make_container(data: pd.DataFrame, toolkit: LinearProductModelToolkit) -> ProductModelDataContainer:
    # Expand/cast each block before concatenation so a float64 numeric ramp cannot
    # promote the entire (mostly one-hot) full-cohort matrix to float64.
    blocks = []
    for name, columns in toolkit.feature_groups.items():
        values = toolkit.preprocess_config[name].transform(data[[name]])
        if hasattr(values, "toarray"):
            values = values.toarray()
        blocks.append(pd.DataFrame(np.asarray(values, dtype=np.float32), columns=columns))
    expanded = pd.concat(blocks, axis=1)
    return ProductModelDataContainer(data, expanded, response=data[TARGET].to_numpy(),
                                     feature_groups=toolkit.feature_groups, as_float32=True)


def prediction_metrics(y, pred, weights) -> dict:
    y, pred, weights = (np.asarray(value, dtype=float) for value in (y, pred, weights))
    finite = np.isfinite(pred)
    counts = {"nonfinite": int((~finite).sum()), "negative": int((pred < 0).sum()),
              "above_one": int((pred > 1).sum()), "zero": int((pred == 0).sum())}
    metrics = {"prediction_range": counts}
    if not finite.all():
        return metrics
    for name, w in (("balance_weighted", weights), ("equal_weight", np.ones(len(y)))):
        actual, expected = float(np.average(y, weights=w)), float(np.average(pred, weights=w))
        # Numerical epsilon is only used in the log metric, never in saved predictions.
        deviance = None
        if not (pred < 0).any():
            safe = np.maximum(pred, 1e-10)
            terms = np.where(y > 0, y * np.log(np.maximum(y, 1e-10) / safe), 0) - y + pred
            deviance = float(np.average(2 * terms, weights=w))
        metrics[name] = {"actual": actual, "predicted": expected,
                         "actual_over_expected": actual / expected if expected > 0 else None,
                         "poisson_deviance": deviance}
    metrics["min_prediction"], metrics["max_prediction"] = float(pred.min()), float(pred.max())
    return metrics


def fit(config: Config, *, smoke_rows: int | None = None) -> dict:
    total_started = time.perf_counter()
    timings = {}
    started = time.perf_counter()
    source = config.output_root / "turnover_frame.parquet"
    frame = pl.read_parquet(source)
    if smoke_rows is not None:
        if smoke_rows < 1:
            raise ValueError("smoke_rows must be positive")
        frame = frame.sample(n=min(smoke_rows, frame.height), seed=42, shuffle=True)
    if frame.is_empty() or not frame[TARGET].sum():
        raise ValueError("Fit needs a nonempty cohort with prepayment events")
    timings["read_frame"] = time.perf_counter() - started
    started = time.perf_counter()
    data = to_model_data(frame)
    timings["model_data"] = time.perf_counter() - started
    print(f"[fit] {len(data):,} rows; constructing main-effect blocks", flush=True)
    started = time.perf_counter()
    toolkit = build_toolkit(data)
    timings["toolkit"] = time.perf_counter() - started
    started = time.perf_counter()
    container = make_container(data, toolkit)
    timings["container"] = time.perf_counter() - started
    print(f"[fit] {container.shape[1]} expanded columns; float32 block arrays", flush=True)
    model = LinearProductRegressorBCD()
    started = time.perf_counter()
    model.fit(container, feature_groups=toolkit.feature_groups, interactions=INTERACTIONS, init_params=None,
              n_iterations=config.n_iterations, early_stopping_rounds=config.early_stopping_rounds,
              ftol=config.ftol, cache_qr_decomp=False, loss="poisson", weights=data["weight"].to_numpy())
    seconds = time.perf_counter() - started
    timings["fit"] = seconds
    started = time.perf_counter()
    pred = model.predict(container)
    timings["predict"] = time.perf_counter() - started
    started = time.perf_counter()
    metrics = prediction_metrics(data[TARGET], pred, data["weight"])
    timings["metrics"] = time.perf_counter() - started
    started = time.perf_counter()
    meta = {"rows": frame.height, "expanded_columns": container.shape[1], "fit_seconds": seconds,
            "loss": "poisson", "weight_source": "c_prev_balance", "target": TARGET,
            "clips": {name: [lower, upper] for name, (lower, upper) in CLIP.items()},
            "clip_counts": {name: int(((frame[name] < lower) | (frame[name] > upper)).sum())
                            for name, (lower, upper) in CLIP.items()},
            "knots": {name: list(knots) for name, knots in KNOTS.items()},
            "categorical": list(CATEGORICAL), "model_inputs": list(MODEL_INPUTS), "interactions": dict(INTERACTIONS),
            "categories": {name: toolkit.preprocess_config[name].categories_[0].tolist() for name in CATEGORICAL},
            "incentive_max": config.incentive_max,
            "n_iterations": config.n_iterations, "early_stopping_rounds": config.early_stopping_rounds,
            "ftol": config.ftol, "smoke_rows": smoke_rows, "frame_sha256": file_sha256(source),
            "global_scalar": float(model.global_scalar_), "best_loss": float(model.best_loss_),
            "best_iteration": int(model.best_iteration_) + 1, "metrics": metrics,
            "actual_sweeps": len(model.loss_history_)}
    predictions = frame.select(*KEYS).with_columns(pl.Series("pred_turnover", pred))
    timings["metadata"] = time.perf_counter() - started
    meta["timing_seconds"] = timings
    started = time.perf_counter()
    with temporary_output(config.output_root / "turnover_predictions.parquet") as path:
        predictions.write_parquet(path, compression="zstd")
        meta["predictions_sha256"] = file_sha256(path)
    with temporary_output(config.output_root / "turnover_model.pkl") as path:
        with path.open("wb") as file:
            pickle.dump({"model": model, "toolkit": toolkit, "meta": meta}, file)
    artifact_seconds = time.perf_counter() - started
    del container, data
    gc.collect()
    total_seconds = time.perf_counter() - total_started
    print(f"[fit] timings(s)={timings}; artifact_write={artifact_seconds:.3f}; "
          f"total={total_seconds:.3f}; sweeps={meta['actual_sweeps']}", flush=True)
    print(f"[fit] block seconds={model.fit_timing_['block_seconds']}", flush=True)
    print(f"[fit] completed in {seconds:.1f}s; metrics={metrics}", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-rows", type=int, help="Explicit diagnostic subset; use a separate output directory")
    args = parser.parse_args()
    fit(read_config(args.config), smoke_rows=args.smoke_rows)


if __name__ == "__main__":
    main()
