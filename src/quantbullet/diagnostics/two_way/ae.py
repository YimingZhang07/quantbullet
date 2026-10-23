"""Binned actual / expected over two variables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from .diagnostic import TwoWayDiagnostic

VALUE_COLUMNS = ("rows", "weight_sum", "actual", "expected", "ae", "ae_se", "actual_mean", "pred_mean")


@dataclass(frozen=True, eq=False)
class TwoWayAE:
    """Actual / expected per (x bin, z bin) cell.

    ``table`` has one row per cell of every observed x bin, in bin order, with empty z bins included
    (``rows`` 0, ``ae`` NaN):

    - ``rows``, ``weight_sum``;
    - ``actual`` = sum(w * y), ``expected`` = sum(w * pred), ``ae`` = actual / expected (NaN when expected <= 0);
    - ``ae_se`` = sqrt(sum(w^2 * (y - pred)^2)) / expected. It assumes no distribution for y and treats rows
      as independent, so it ignores the correlation between rows of one loan;
    - ``actual_mean`` and ``pred_mean``, the weighted means.

    ``x_bin`` / ``z_bin`` hold ``pd.Interval`` (or values for ``BinSpec.round``); ``x_levels`` / ``z_levels``
    list every bin in order. ``excluded`` adds ``outside_bins`` to the diagnostic's exclusions.
    """

    x: str
    z: str
    x_scale: str
    z_scale: str
    labels: dict
    x_levels: tuple
    z_levels: tuple
    table: pd.DataFrame
    n_obs: int
    excluded: dict
    x_range: tuple[float, float]
    z_range: tuple[float, float]

    def label(self, name: str) -> str:
        return self.labels.get(name, name)

    def pivot(self, value: str = "ae") -> pd.DataFrame:
        """``value`` as a wide table: x bins down the index, z bins across the columns."""
        if value not in VALUE_COLUMNS:
            raise ValueError(f"value must be one of {VALUE_COLUMNS}")
        x_index = pd.Index(list(self.x_levels), dtype=object, name=self.label(self.x))
        z_index = pd.Index(list(self.z_levels), dtype=object, name=self.label(self.z))
        values = np.full((len(x_index), len(z_index)), np.nan)
        rows = [x_index.get_loc(level) for level in self.table["x_bin"]]
        cols = [z_index.get_loc(level) for level in self.table["z_bin"]]
        values[rows, cols] = self.table[value].to_numpy(dtype=float)
        return pd.DataFrame(values, index=x_index, columns=z_index)

    def plot(self, **kwargs):
        """Heatmap of A/E above one slice per x bin; see ``quantbullet.plot.draw_two_way_ae``."""
        from quantbullet.plot.two_way import draw_two_way_ae
        return draw_two_way_ae(self, **kwargs)


def summarize_ae(diag: TwoWayDiagnostic, *, x_bins, z_bins) -> TwoWayAE:
    """Aggregate the diagnostic's rows once with ``summarize_binned_means``: z as x, x as col."""
    from quantbullet.plot.binned_means import BinSpec, summarize_binned_means

    for name, spec in (("x_bins", x_bins), ("z_bins", z_bins)):
        if not isinstance(spec, BinSpec):
            raise TypeError(f"{name} must be a BinSpec")
    weighted = diag.weight is not None
    # With weights w, the weighted sum of w * (y - pred)^2 is sum(w^2 * (y - pred)^2).
    residual = (pl.col("_y") - pl.col("_pred")) ** 2
    frame = diag._frame.with_columns((pl.col("_weight") * residual if weighted else residual).alias("_sq"))
    data = summarize_binned_means(frame, x="_z", y=["_y", "_pred", "_sq"], weight="_weight" if weighted else None,
                                  col="_x", bins={"_x": x_bins, "_z": z_bins})
    summary = data.summary
    weight_sum = summary["weight_sum"].to_numpy(dtype=float)
    actual = summary["_y__weighted_sum"].to_numpy(dtype=float)
    expected = summary["_pred__weighted_sum"].to_numpy(dtype=float)
    squares = summary["_sq__weighted_sum"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        positive = expected > 0
        has_weight = weight_sum > 0
        table = pd.DataFrame({
            "x_bin": summary["col"].to_numpy(dtype=object),
            "z_bin": summary["x"].to_numpy(dtype=object),
            "rows": summary["count"].to_numpy(dtype=np.int64),
            "weight_sum": weight_sum,
            "actual": actual,
            "expected": expected,
            "ae": np.where(positive, actual / expected, np.nan),
            "ae_se": np.where(positive, np.sqrt(squares) / expected, np.nan),
            "actual_mean": np.where(has_weight, actual / weight_sum, np.nan),
            "pred_mean": np.where(has_weight, expected / weight_sum, np.nan),
        })
    x_levels, z_levels = tuple(data.levels["col"]), tuple(data.levels["x"])
    order = np.lexsort(([z_levels.index(v) for v in table["z_bin"]], [x_levels.index(v) for v in table["x_bin"]]))
    excluded = dict(diag.excluded)
    if data.excluded_count:
        excluded["outside_bins"] = int(data.excluded_count)
    return TwoWayAE(x=diag.x, z=diag.z, x_scale=diag.x_scale, z_scale=diag.z_scale, labels=dict(diag.labels),
                    x_levels=x_levels, z_levels=z_levels, table=table.iloc[order].reset_index(drop=True),
                    n_obs=diag.n_obs - int(data.excluded_count), excluded=excluded,
                    x_range=diag.x_range, z_range=diag.z_range)
