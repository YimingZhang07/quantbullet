"""Bin specifications and the Polars key expressions that apply them.

Each grouping dimension becomes one key expression: explicit or fitted
interval edges, rounding to a width, declared category codes, or the raw
values. Edges are fitted once on the whole column, never per facet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import polars as pl


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


def _is_datetime(dtype: pl.DataType) -> bool:
    return dtype == pl.Date or isinstance(dtype, pl.Datetime)


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

