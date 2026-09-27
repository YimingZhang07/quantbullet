"""Backend-independent, vectorized statistics for grouped mean plots.

Only selected columns are extracted from pandas/Polars. NumPy performs the
aggregation, so both inputs follow exactly the same rules without PyArrow.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinSpec:
    """Explicit binning: finite edges, global quantiles, or fixed-width bins.

    All intervals are right-closed and include the lowest boundary. ``step``
    uses multiples of its width as edges (not rounding to the nearest value).
    Missing/nonfinite values and values outside explicit edges are excluded.
    """

    method: Literal["edges", "quantile", "step"]
    breaks: tuple[float, ...] = ()
    n_bins: int = 10
    width: float | None = None

    def __post_init__(self):
        if self.method == "edges":
            edges = np.asarray(self.breaks, dtype=float)
            if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
                raise ValueError("edges must contain at least two finite, strictly increasing values")
            object.__setattr__(self, "breaks", tuple(edges))
        elif self.method == "quantile":
            if isinstance(self.n_bins, bool) or not isinstance(self.n_bins, (int, np.integer)) or self.n_bins < 1:
                raise ValueError("n_bins must be a positive integer")
        elif self.method == "step":
            if self.width is None or not np.isfinite(self.width) or self.width <= 0:
                raise ValueError("width must be finite and positive")
        else:
            raise ValueError("unknown binning method")

    @classmethod
    def edges(cls, values: Sequence[float]) -> BinSpec:
        return cls("edges", breaks=tuple(values))

    @classmethod
    def quantile(cls, n_bins: int = 10) -> BinSpec:
        return cls("quantile", n_bins=n_bins)

    @classmethod
    def step(cls, width: float) -> BinSpec:
        return cls("step", width=width)


@dataclass
class GroupedMeansData:
    """Reusable aggregation; role columns in ``summary`` are x/group/row/col.

    For each metric, columns ``<metric>__mean``, ``<metric>__valid_count`` and
    ``<metric>__weight_sum`` describe its independently filtered observations.
    ``count`` counts all rows with valid grouping keys, including missing y or
    weights. Zero weights are valid observations but contribute no weight.
    Empty x bins are retained for every observed combination of other roles.
    """

    summary: pd.DataFrame
    bin_info: dict[str, dict]
    dimensions: dict[str, str]
    metrics: tuple[str, ...]
    levels: dict[str, tuple]
    x_positions: np.ndarray
    x_widths: np.ndarray
    input_count: int
    excluded_count: int


def _column(df, name: str) -> pd.Series:
    # Preserve pandas categorical order, and avoid Polars.to_pandas()/PyArrow.
    values = df[name]
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(values.to_numpy())


def _numeric(series: pd.Series, name: str) -> np.ndarray:
    if series.isna().all():
        return np.full(len(series), np.nan)
    if not pd.api.types.is_numeric_dtype(series) and not series.empty:
        raise TypeError(f"{name!r} must be numeric")
    return series.to_numpy(dtype=float, na_value=np.nan)


def _encode(series: pd.Series, spec: BinSpec | None, name: str):
    if spec is None:
        if pd.api.types.is_numeric_dtype(series):
            series = series.where(np.isfinite(_numeric(series, name)))
        if isinstance(series.dtype, pd.CategoricalDtype):
            levels = tuple(series.cat.categories)
            codes = series.cat.codes.to_numpy()
        else:
            codes, uniques = pd.factorize(series, sort=True)
            levels = tuple(uniques)
        categorical = isinstance(series.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(series)
        return codes, levels, {"method": "discrete", "edges": None, "categorical": categorical}

    values = _numeric(series, name)
    finite = values[np.isfinite(values)]
    if spec.method == "edges":
        edges = np.asarray(spec.breaks)
    elif not finite.size:
        edges = np.array([], dtype=float)
    elif spec.method == "quantile":
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, spec.n_bins + 1)))
    else:
        low = np.floor(finite.min() / spec.width)
        high = max(np.ceil(finite.max() / spec.width), low + 1)
        edges = np.arange(low, high + 1) * spec.width

    if len(edges) == 1:  # A constant column still has one usable quantile bin.
        levels = (pd.Interval(edges[0], edges[0], closed="both"),)
        codes = np.where(values == edges[0], 0, -1)
    elif len(edges) >= 2:
        levels = tuple(pd.Interval(a, b, closed="both" if i == 0 else "right") for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])))
        codes = np.searchsorted(edges, values, side="left") - 1
        codes[values == edges[0]] = 0
        codes[~np.isfinite(values) | (values < edges[0]) | (values > edges[-1])] = -1
    else:
        codes = np.full(len(series), -1, dtype=int)
        levels = ()
    return codes, levels, {
        "method": spec.method, "edges": tuple(edges),
        "closed": "right", "include_lowest": True,
    }


def summarize_grouped_means(
    df,
    *,
    x: str,
    y: str | Sequence[str],
    weight: str | None = None,
    group: str | None = None,
    row: str | None = None,
    col: str | None = None,
    bins: Mapping[str, BinSpec] | None = None,
) -> GroupedMeansData:
    """Aggregate pandas or eager Polars data without mutating the input.

    ``x`` is required. At most two of group/row/col may be supplied. Binning
    specs are keyed by source column and fitted globally before group-by.
    Negative weights raise; nonfinite weights/y are omitted per metric.
    Missing grouping keys are excluded and counted in ``excluded_count``.
    Ordered pandas categoricals preserve category order; other values sort.
    """
    if not isinstance(df, pd.DataFrame):
        try:
            import polars as pl
        except ImportError:
            pl = None
        if pl is None or not isinstance(df, pl.DataFrame):
            raise TypeError("df must be a pandas or eager Polars DataFrame")
    metrics = (y,) if isinstance(y, str) else tuple(y)
    if not metrics or any(not isinstance(m, str) for m in metrics) or len(set(metrics)) != len(metrics):
        raise ValueError("y must contain one or more distinct column names")
    dimensions = {role: name for role, name in (("x", x), ("group", group), ("row", row), ("col", col)) if name is not None}
    if x is None or len(dimensions) > 3:
        raise ValueError("supply x and at most two of group, row, col")
    if len(set(dimensions.values())) != len(dimensions):
        raise ValueError("each dimension must use a distinct column")
    bins = dict(bins or {})
    if set(bins) - set(dimensions.values()):
        raise ValueError("bins keys must name grouping dimension columns")
    if any(not isinstance(spec, BinSpec) for spec in bins.values()):
        raise TypeError("bins values must be BinSpec instances")
    required = set(dimensions.values()) | set(metrics) | ({weight} if weight is not None else set())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if isinstance(df, pd.DataFrame) and not df.columns.is_unique:
        raise ValueError("input column names must be unique")

    n = len(df)
    weights = np.ones(n) if weight is None else _numeric(_column(df, weight), weight)
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    codes, levels, bin_info = {}, {}, {}
    for role, name in dimensions.items():
        codes[role], levels[role], info = _encode(_column(df, name), bins.get(name), name)
        bin_info[name] = {**info, "levels": levels[role]}
    roles = list(dimensions)
    valid_keys = np.logical_and.reduce([codes[role] >= 0 for role in roles])
    key_array = np.column_stack([codes[role][valid_keys] for role in roles])
    observed, inverse = np.unique(key_array, axis=0, return_inverse=True)
    n_observed = len(observed)
    stats = pd.DataFrame(observed, columns=roles)
    stats["count"] = np.bincount(inverse, minlength=n_observed)
    w = weights[valid_keys]
    for metric in metrics:
        values = _numeric(_column(df, metric), metric)[valid_keys]
        valid = np.isfinite(values) & np.isfinite(w)
        count = np.bincount(inverse[valid], minlength=n_observed)
        weight_sum = np.bincount(inverse[valid], weights=w[valid], minlength=n_observed)
        weighted_sum = np.bincount(inverse[valid], weights=w[valid] * values[valid], minlength=n_observed)
        mean = np.full(n_observed, np.nan)
        np.divide(weighted_sum, weight_sum, out=mean, where=weight_sum > 0)
        stats[f"{metric}__mean"] = mean
        stats[f"{metric}__valid_count"] = count
        stats[f"{metric}__weight_sum"] = weight_sum

    # Complete x only within observed contexts, avoiding a full row-level
    # Cartesian product of sparse/high-cardinality grouping dimensions.
    contexts = np.unique(observed[:, 1:], axis=0) if n_observed else np.empty((0, len(roles) - 1), dtype=int)
    if len(roles) == 1:
        contexts = [()]
    complete = [(xi, *context) for context, xi in product(contexts, range(len(levels["x"])))]
    grid = pd.DataFrame(complete, columns=roles, dtype=int)
    summary = grid.merge(stats, on=roles, how="left", sort=False)
    for field in ["count"] + [f"{m}__valid_count" for m in metrics]:
        summary[field] = summary[field].fillna(0).astype(np.int64)
    for metric in metrics:
        summary[f"{metric}__weight_sum"] = summary[f"{metric}__weight_sum"].fillna(0)
    for role in roles:
        summary[role] = pd.Categorical.from_codes(summary[role].to_numpy(dtype=int), categories=list(levels[role]), ordered=True)

    x_levels = levels["x"]
    if x in bins:
        positions = np.array([interval.mid for interval in x_levels], dtype=float)
        widths = np.array([interval.length for interval in x_levels], dtype=float)
        widths[widths == 0] = 1.0
    elif x_levels and not bin_info[x]["categorical"]:
        positions = np.array(x_levels, dtype=float)
        widths = np.full(len(positions), np.min(np.diff(positions)) if len(positions) > 1 else 1.0)
    else:
        positions = np.arange(len(x_levels), dtype=float)
        widths = np.ones(len(x_levels))
    return GroupedMeansData(summary, bin_info, dimensions, metrics, levels, positions, widths, n, int((~valid_keys).sum()))
