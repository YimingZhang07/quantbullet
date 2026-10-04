"""Binning, dimension metadata and plot-ready summaries for grouped means.

Raw rows are processed once: the selected columns form a slim Polars frame,
each dimension becomes a Polars key expression, and a single group-by
produces the statistics. Level ordering, empty-bin completion and plot
positions are derived from that small aggregated table only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl

from quantbullet.utils.grouped_stats import grouped_weighted_summary

_STAT_SUFFIXES = ("valid_count", "weight_sum", "weighted_sum", "mean")


@dataclass(frozen=True)
class BinSpec:
    """Explicit binning: edges, quantiles, fixed-width intervals or rounding.

    All intervals are right-closed and include the lowest boundary. ``step``
    uses multiples of its width as edges (not rounding to the nearest value).
    Missing/nonfinite values and values outside explicit edges are excluded.
    """

    method: Literal["edges", "quantile", "step", "round"]
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
        elif self.method in {"step", "round"}:
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

    @classmethod
    def round(cls, width: float) -> BinSpec:
        """Round to the nearest width multiple (ties to even), not intervals."""
        return cls("round", width=width)


@dataclass
class GroupedMeansData:
    """Reusable aggregation; role columns in ``summary`` are x/group/row/col.

    For each metric, ``__mean``, ``__valid_count``, ``__weight_sum`` and
    ``__weighted_sum`` describe its independently filtered observations.
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

    @classmethod
    def from_summary(
        cls, summary: pd.DataFrame, *, x: str,
        mean_columns: Mapping[str, str], count: str = "count", col: str | None = None,
    ) -> GroupedMeansData:
        """Adapt a small preaggregated table without regrouping raw observations.

        Use this for bin-level estimates that are not weighted means, such as
        implied-actual ratios. ``mean_columns`` maps metric names to supplied
        estimate columns. Rows with missing keys count toward ``excluded_count``;
        metric weight sums and valid counts are not inferred.
        """
        if not mean_columns:
            raise ValueError("mean_columns must name at least one metric")
        dimensions = {"x": x} if col is None else {"x": x, "col": col}
        missing = {*dimensions.values(), count, *mean_columns.values()} - set(summary.columns)
        if missing:
            raise ValueError(f"missing summary columns: {sorted(missing)}")
        counts = summary[count].to_numpy(dtype=float)
        if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
            raise ValueError("summary counts must be finite nonnegative integers")
        counts = counts.astype(np.int64)
        codes, levels, bin_info = {}, {}, {}
        for role, name in dimensions.items():
            codes[role], levels[role], info = _encode_small(summary[name].reset_index(drop=True))
            bin_info[name] = {**info, "levels": levels[role]}
        valid = np.logical_and.reduce([codes[role] >= 0 for role in dimensions])
        stats = pd.DataFrame({role: codes[role][valid] for role in dimensions})
        stats["count"] = counts[valid]
        for metric, source in mean_columns.items():
            stats[f"{metric}__mean"] = summary[source].to_numpy(dtype=float)[valid]
        if stats.duplicated(list(dimensions)).any():
            raise ValueError("summary contains duplicate x/col keys")
        return _assemble(stats, dimensions, tuple(mean_columns), levels, bin_info,
                         int(counts.sum()), int(counts[~valid].sum()))

    def map_means(self, transform: Callable, metrics: Sequence[str] | None = None) -> GroupedMeansData:
        """Return a copy with ``transform`` applied to aggregated means only.

        Use this for display units after weighting, e.g. bin-level SMM -> CPR.
        Weighted sums keep their original units; this object is unchanged.
        """
        selected = self.metrics if metrics is None else tuple(metrics)
        unknown = set(selected) - set(self.metrics)
        if unknown:
            raise ValueError(f"unknown metrics: {sorted(unknown)}")
        summary = self.summary.copy()
        for metric in selected:
            column = f"{metric}__mean"
            summary[column] = np.asarray(transform(summary[column]), dtype=float)
        return replace(self, summary=summary)

    def mask_support(self, min_count: int) -> GroupedMeansData:
        """Return a copy whose means are missing where ``count < min_count``.

        Counts are kept, so low-support bins still show their bars while the
        curves break there. This object is unchanged.
        """
        if isinstance(min_count, bool) or not isinstance(min_count, (int, np.integer)) or min_count < 0:
            raise ValueError("min_count must be a nonnegative integer")
        summary = self.summary.copy()
        low = summary["count"].to_numpy() < min_count
        for metric in self.metrics:
            summary.loc[low, f"{metric}__mean"] = np.nan
        return replace(self, summary=summary)


def _is_datetime(dtype: pl.DataType) -> bool:
    return dtype == pl.Date or isinstance(dtype, pl.Datetime)


def _numeric(series: pd.Series, name: str) -> np.ndarray:
    if series.isna().all():
        return np.full(len(series), np.nan)
    if not pd.api.types.is_numeric_dtype(series) and not series.empty:
        raise TypeError(f"{name!r} must be numeric")
    return series.to_numpy(dtype=float, na_value=np.nan)


def _pandas_values(series: pd.Series, name: str) -> pl.Series:
    """Convert one pandas key column through NumPy, without PyArrow."""
    dtype = series.dtype
    if isinstance(dtype, np.dtype) and dtype.kind in "biufmM":
        return pl.Series(name, series.to_numpy())
    if pd.api.types.is_numeric_dtype(series) and hasattr(dtype, "numpy_dtype"):
        # Nullable extension arrays keep their value dtype; NA becomes null.
        values = pl.Series(name, series.to_numpy(dtype=dtype.numpy_dtype, na_value=0))
        missing = series.isna().to_numpy()
        if missing.any():
            values = pl.select(pl.when(pl.Series(missing)).then(None).otherwise(values).alias(name)).to_series()
        return values
    # Python objects (strings, dates); Polars infers String/Date/Datetime.
    return pl.Series(name, series.to_numpy(dtype=object, na_value=None).tolist())


def _slim_frame(df, dimensions: Mapping[str, str], metrics: Sequence[str], weight: str | None,
                bins: Mapping[str, BinSpec]) -> tuple[pl.DataFrame, dict[str, tuple]]:
    """Select required columns into Polars under internal names.

    Dimensions become ``_d_<role>``, metrics ``_metric<i>`` and the weight
    ``_weight`` (all Float64 for metrics/weight). Ordered categories (pandas
    categorical, Polars Enum) are stored as codes; their declared levels are
    returned by role so empty categories survive aggregation.
    """
    declared: dict[str, tuple] = {}
    values = [*metrics, *([weight] if weight is not None else [])]
    if isinstance(df, pl.DataFrame):
        for name in values:
            dtype = df.schema[name]
            if not (dtype.is_numeric() or dtype in (pl.Boolean, pl.Null)):
                raise TypeError(f"{name!r} must be numeric")
        expressions = []
        for role, name in dimensions.items():
            column = pl.col(name)
            dtype = df.schema[name]
            if isinstance(dtype, pl.Enum) and name not in bins:
                declared[role] = tuple(dtype.categories.to_list())
                column = column.to_physical()
            elif name in bins and not (dtype.is_numeric() or dtype == pl.Null):
                raise TypeError(f"{name!r} must be numeric")
            expressions.append(column.alias(f"_d_{role}"))
        expressions += [pl.col(m).cast(pl.Float64).alias(f"_metric{i}") for i, m in enumerate(metrics)]
        if weight is not None:
            expressions.append(pl.col(weight).cast(pl.Float64).alias("_weight"))
        return df.select(expressions), declared

    columns = {}
    for role, name in dimensions.items():
        series = df[name]
        if name in bins:
            columns[f"_d_{role}"] = _numeric(series, name)
        elif isinstance(series.dtype, pd.CategoricalDtype):
            declared[role] = tuple(series.cat.categories)
            columns[f"_d_{role}"] = series.cat.codes.to_numpy()
        else:
            columns[f"_d_{role}"] = _pandas_values(series, f"_d_{role}")
    for i, metric in enumerate(metrics):
        columns[f"_metric{i}"] = _numeric(df[metric], metric)
    if weight is not None:
        columns["_weight"] = _numeric(df[weight], weight)
    return pl.DataFrame(columns), declared


def _fit_edges(frame: pl.DataFrame, values: pl.Expr, spec: BinSpec) -> np.ndarray:
    """Global edges from finite values; one small select, no row-level copy."""
    if spec.method == "edges":
        return np.asarray(spec.breaks, dtype=float)
    finite = values.filter(values.is_finite())
    if spec.method == "quantile":
        probabilities = np.linspace(0, 1, spec.n_bins + 1)
        row = frame.select([finite.quantile(float(p), "linear").alias(f"q{i}")
                            for i, p in enumerate(probabilities)]).row(0)
        return np.array([], dtype=float) if row[0] is None else np.unique(np.asarray(row, dtype=float))
    low, high = frame.select(finite.min().alias("low"), finite.max().alias("high")).row(0)
    if low is None:
        return np.array([], dtype=float)
    low = np.floor(low / spec.width)
    high = max(np.ceil(high / spec.width), low + 1)
    return np.arange(low, high + 1) * spec.width


def _dimension_key(frame: pl.DataFrame, role: str, spec: BinSpec | None, declared: tuple | None):
    """Return a key expression (null means excluded), known levels and info.

    Known levels mean the key holds integer codes into them (declared
    categories and interval bins); otherwise levels come from observed keys.
    """
    column = f"_d_{role}"
    values = pl.col(column)
    dtype = frame.schema[column]
    if declared is not None:
        return (pl.when(values >= 0).then(values).alias(role), declared,
                {"method": "discrete", "edges": None, "categorical": True, "datetime": False})
    if spec is None:
        if dtype.is_float():
            values = pl.when(values.is_finite()).then(values)
        datetime = _is_datetime(dtype)
        categorical = not (dtype.is_numeric() or dtype == pl.Boolean or datetime)
        return values.alias(role), None, {"method": "discrete", "edges": None,
                                          "categorical": categorical, "datetime": datetime}

    values = values.cast(pl.Float64)
    finite = values.is_finite()
    if spec.method == "round":
        key = pl.when(finite).then((values / spec.width).round() * spec.width)
        return key.alias(role), None, {"method": "round", "width": spec.width,
                                       "edges": None, "categorical": False}
    edges = _fit_edges(frame, values, spec)
    if len(edges) >= 2:
        levels = tuple(pd.Interval(a, b, closed="both" if i == 0 else "right")
                       for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])))
        position = pl.lit(pl.Series(edges)).search_sorted(values, side="left").cast(pl.Int64)
        in_range = finite & (values >= edges[0]) & (values <= edges[-1])
        key = pl.when(in_range).then(pl.max_horizontal(position - 1, 0))  # lowest edge -> first bin
    elif len(edges) == 1:  # A constant column still has one usable quantile bin.
        levels = (pd.Interval(edges[0], edges[0], closed="both"),)
        key = pl.when(finite & (values == edges[0])).then(pl.lit(0, dtype=pl.Int64))
    else:
        levels, key = (), pl.lit(None, dtype=pl.Int64)
    return key.alias(role), levels, {"method": spec.method, "edges": tuple(edges),
                                     "closed": "right", "include_lowest": True}


def _observed_codes(keys: pl.Series) -> tuple[np.ndarray, tuple]:
    """Sort the distinct keys of the small aggregated table into levels."""
    if keys.dtype == pl.Categorical:
        keys = keys.cast(pl.String)
    values = keys.to_list()
    if keys.dtype.is_float():
        values = [0.0 if value == 0 else value for value in values]  # -0.0 is not its own level
    codes, levels = pd.factorize(pd.Series(values, dtype=object), sort=True)
    return codes, tuple(levels)


def _encode_small(series: pd.Series) -> tuple[np.ndarray, tuple, dict]:
    """Encode the key column of a small preaggregated pandas table."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return (series.cat.codes.to_numpy(dtype=np.int64), tuple(series.cat.categories),
                {"method": "preaggregated", "edges": None, "categorical": True, "datetime": False})
    numeric = pd.api.types.is_numeric_dtype(series)
    if numeric:
        series = series.where(np.isfinite(series.to_numpy(dtype=float, na_value=np.nan)))
    datetime = not numeric and (pd.api.types.is_datetime64_any_dtype(series)
                                or pd.api.types.infer_dtype(series.dropna()) in {"date", "datetime", "datetime64"})
    codes, uniques = pd.factorize(series, sort=True)
    return codes, tuple(uniques), {"method": "preaggregated", "edges": None,
                                   "categorical": not numeric and not datetime, "datetime": datetime}


def _assemble(stats: pd.DataFrame, dimensions: Mapping[str, str], metrics: tuple[str, ...],
              levels: dict[str, tuple], bin_info: dict[str, dict],
              input_count: int, excluded_count: int) -> GroupedMeansData:
    """Complete empty x bins, decode role codes and place x; small tables only.

    ``stats`` holds integer role codes into ``levels`` plus ``count`` and
    metric statistics. Shared by raw-data and preaggregated summaries.
    """
    roles = list(dimensions)
    # Complete x only within observed contexts, avoiding a Cartesian product
    # of sparse/high-cardinality grouping dimensions.
    contexts = (stats[roles[1:]].drop_duplicates().sort_values(roles[1:]).to_numpy()
                if len(roles) > 1 else [()])
    complete = [(xi, *context) for context, xi in product(contexts, range(len(levels["x"])))]
    grid = pd.DataFrame(complete, columns=roles, dtype=np.int64)
    summary = grid.merge(stats.astype({role: np.int64 for role in roles}), on=roles, how="left", sort=False)
    summary["count"] = summary["count"].fillna(0).astype(np.int64)
    for metric in metrics:
        for suffix in ("valid_count", "weight_sum", "weighted_sum"):
            column = f"{metric}__{suffix}"
            if column in summary:
                summary[column] = summary[column].fillna(0)
                if suffix == "valid_count":
                    summary[column] = summary[column].astype(np.int64)
    for role in roles:
        summary[role] = pd.Categorical.from_codes(summary[role].to_numpy(dtype=int),
                                                  categories=list(levels[role]), ordered=True)

    x_levels = levels["x"]
    info = bin_info[dimensions["x"]]
    if info.get("edges") is not None:  # interval bins
        positions = np.array([interval.mid for interval in x_levels], dtype=float)
        widths = np.array([interval.length for interval in x_levels], dtype=float)
        widths[widths == 0] = 1.0
    elif info.get("datetime", False):
        from matplotlib.dates import date2num
        positions = np.asarray(date2num(pd.to_datetime(list(x_levels)).to_pydatetime()), dtype=float)
        widths = np.full(len(positions), np.min(np.diff(positions)) if len(positions) > 1 else 1.0)
    elif x_levels and not info.get("categorical", False):
        positions = np.array(x_levels, dtype=float)
        width = info.get("width") or (np.min(np.diff(positions)) if len(positions) > 1 else 1.0)
        widths = np.full(len(positions), width)
    else:
        positions = np.arange(len(x_levels), dtype=float)
        widths = np.ones(len(x_levels))
    return GroupedMeansData(summary, bin_info, dict(dimensions), metrics, levels,
                            positions, widths, input_count, excluded_count)


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
    Ordered categories (pandas categorical, Polars Enum) keep their declared
    order, including empty categories; other values sort.
    """
    if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
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

    frame, declared = _slim_frame(df, dimensions, metrics, weight, bins)
    roles = list(dimensions)
    keys, levels, bin_info = [], {}, {}
    for role, name in dimensions.items():
        key, levels[role], info = _dimension_key(frame, role, bins.get(name), declared.get(role))
        keys.append(key)
        bin_info[name] = info
    internal = [f"_metric{i}" for i in range(len(metrics))]
    slim = frame.select(keys + internal + (["_weight"] if weight is not None else []))
    stats = grouped_weighted_summary(slim, by=roles, metrics=internal,
                                     weight="_weight" if weight is not None else None)

    # Everything below runs on the aggregated table (bins x contexts rows).
    has_keys = pl.all_horizontal([pl.col(role).is_not_null() for role in roles])
    excluded = int(stats.filter(~has_keys)["count"].sum())
    stats = stats.filter(has_keys)
    table = {}
    for role in roles:
        if levels[role] is None:
            table[role], levels[role] = _observed_codes(stats[role])
        else:
            table[role] = stats[role].cast(pl.Int64).to_numpy()
        bin_info[dimensions[role]]["levels"] = levels[role]
    table["count"] = stats["count"].to_numpy()
    for i, metric in enumerate(metrics):
        for suffix in _STAT_SUFFIXES:
            table[f"{metric}__{suffix}"] = stats[f"_metric{i}__{suffix}"].to_numpy()
    return _assemble(pd.DataFrame(table), dimensions, metrics, levels, bin_info, len(df), excluded)
