"""General grouped statistics, independent of plotting and binning."""
from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def grouped_weighted_summary(
    df: pl.DataFrame,
    *,
    by: Sequence[str],
    metrics: Sequence[str],
    weight: str | None = None,
) -> pl.DataFrame:
    """Count rows and summarize each metric's finite observations independently.

    Null/nonfinite metrics or weights do not contribute to that metric's
    statistics. Zero weights count as valid observations; zero total weight
    produces a null mean. Negative weights raise, including on excluded rows.
    Group keys (including null keys) are passed through unchanged. Only the
    small aggregated result is sorted, never the input observations.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError("df must be an eager Polars DataFrame")
    by, metrics = tuple(by), tuple(metrics)
    if not by or len(set(by)) != len(by):
        raise ValueError("by must contain distinct grouping column names")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must contain distinct metric column names")
    required = set(by) | set(metrics) | ({weight} if weight is not None else set())
    if missing := required - set(df.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    generated = {"count"} | {
        f"{m}__{suffix}" for m in metrics
        for suffix in ("valid_count", "weight_sum", "weighted_sum", "mean")
    }
    if generated & set(by):
        raise ValueError("grouping names conflict with summary columns")
    for name in set(metrics) | ({weight} if weight is not None else set()):
        if not df.schema[name].is_numeric() and df.schema[name] != pl.Null:
            raise TypeError(f"{name!r} must be numeric")
    w = pl.lit(1.0) if weight is None else pl.col(weight).cast(pl.Float64)
    if weight is not None and df.select((w < 0).any()).item():
        raise ValueError("weights must be nonnegative")

    expressions = [pl.len().cast(pl.Int64).alias("count")]
    for metric in metrics:
        value = pl.col(metric).cast(pl.Float64)
        valid = (value.is_finite() & w.is_finite()).fill_null(False)
        expressions.extend([
            valid.sum().cast(pl.Int64).alias(f"{metric}__valid_count"),
            pl.when(valid).then(w).otherwise(0.0).sum().alias(f"{metric}__weight_sum"),
            pl.when(valid).then(w * value).otherwise(0.0).sum().alias(f"{metric}__weighted_sum"),
        ])
    return df.group_by(list(by)).agg(expressions).with_columns([
        pl.when(pl.col(f"{m}__weight_sum") > 0)
        .then(pl.col(f"{m}__weighted_sum") / pl.col(f"{m}__weight_sum"))
        .otherwise(None).alias(f"{m}__mean")
        for m in metrics
    ]).sort(list(by))
