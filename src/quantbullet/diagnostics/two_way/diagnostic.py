"""``TwoWayDiagnostic``: the five input columns, read and checked once for both methods."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from quantbullet.plot.binned_means import BinSpec

    from .ae import TwoWayAE
    from .correction import TwoWayCorrection

Scale = Literal["linear", "log"]
SCALES = ("linear", "log")


def _float_column(df, name: str) -> pl.Series:
    """One input column as Float64, through NumPy for pandas (no PyArrow); nulls stay null."""
    if isinstance(df, pl.DataFrame):
        dtype = df.schema[name]
        if not (dtype.is_numeric() or dtype in (pl.Boolean, pl.Null)):
            raise TypeError(f"{name!r} must be numeric")
        return df[name].cast(pl.Float64)
    series = df[name]
    if not (pd.api.types.is_numeric_dtype(series) or series.isna().all()):
        raise TypeError(f"{name!r} must be numeric")
    return pl.Series(series.to_numpy(dtype=float, na_value=np.nan))


class TwoWayDiagnostic:
    """Rows of (x, z, y, pred, weight) for a two-way look at a model's predictions.

    ``df`` is pandas or eager Polars and is not modified; only the five columns are kept, as Float64. ``pred`` is
    the model's prediction for each row (a column, not a model). ``x`` and ``z`` are the two variables: x runs
    down the heatmap rows and splits the slice panels, z runs across the columns and along each slice.
    ``x_scale`` / ``z_scale`` describe the variables themselves: ``"log"`` places a variable on a log axis in
    plots, and ``correction`` interpolates between its knots in log space.

    Rows are excluded, and counted in ``excluded`` by reason, when any of the five values is null or nonfinite
    (``missing``) or a log-scale variable is not positive (``nonpositive_x`` / ``nonpositive_z``). Negative
    weights raise. ``ae`` and ``correction`` check what they need of y and pred themselves.
    """

    def __init__(self, df, *, x: str, z: str, y: str, pred: str, weight: str | None = None,
                 x_scale: Scale = "linear", z_scale: Scale = "linear", labels: Mapping[str, str] | None = None):
        if not isinstance(df, (pd.DataFrame, pl.DataFrame)):
            raise TypeError("df must be a pandas or eager Polars DataFrame")
        names = {"x": x, "z": z, "y": y, "pred": pred, **({"weight": weight} if weight is not None else {})}
        if any(not isinstance(name, str) for name in names.values()):
            raise TypeError("column names must be strings")
        if len({x, z, y, pred}) != 4:   # weight may reuse a column, e.g. weighting by balance with x = balance
            raise ValueError("x, z, y and pred must name distinct columns")
        if isinstance(df, pd.DataFrame) and not df.columns.is_unique:
            raise ValueError("input column names must be unique")
        if missing := sorted(set(names.values()) - set(df.columns)):
            raise ValueError(f"missing columns: {missing}")
        for role, scale in (("x_scale", x_scale), ("z_scale", z_scale)):
            if scale not in SCALES:
                raise ValueError(f"{role} must be 'linear' or 'log'")

        frame = pl.DataFrame({f"_{role}": _float_column(df, name) for role, name in names.items()})
        finite = frame.select(pl.all_horizontal(
            [pl.col(column).is_finite().fill_null(False) for column in frame.columns]).alias("finite"))["finite"]
        excluded = {}
        if dropped := int((~finite).sum()):
            excluded["missing"] = dropped
        frame = frame.filter(finite)
        for axis, scale in (("x", x_scale), ("z", z_scale)):
            if scale == "log":
                positive = frame[f"_{axis}"] > 0
                if dropped := int((~positive).sum()):
                    excluded[f"nonpositive_{axis}"] = dropped
                frame = frame.filter(positive)
        if weight is not None and (frame["_weight"] < 0).any():
            raise ValueError("weights must be nonnegative")
        if frame.is_empty():
            raise ValueError("no rows left after excluding missing values")

        self.x, self.z, self.y, self.pred, self.weight = x, z, y, pred, weight
        self.x_scale, self.z_scale = x_scale, z_scale
        self.labels = dict(labels or {})
        self.input_rows = len(df)
        self.n_obs = frame.height
        self.excluded = excluded
        self.x_range = (float(frame["_x"].min()), float(frame["_x"].max()))
        self.z_range = (float(frame["_z"].min()), float(frame["_z"].max()))
        self._frame = frame

    def __repr__(self):
        return (f"TwoWayDiagnostic(x={self.x!r}, z={self.z!r}, y={self.y!r}, pred={self.pred!r}, "
                f"weight={self.weight!r}, n_obs={self.n_obs}, excluded={self.excluded})")

    def _array(self, role: str) -> np.ndarray:
        """One kept column as a float64 array; unit weights when unweighted."""
        if role == "weight" and self.weight is None:
            return np.ones(self.n_obs)
        return self._frame[f"_{role}"].to_numpy()

    def ae(self, *, x_bins: BinSpec, z_bins: BinSpec) -> TwoWayAE:
        """Actual / expected per (x bin, z bin) cell; see ``TwoWayAE``.

        ``x_bins`` and ``z_bins`` are ``BinSpec`` (edges, quantile, step or round). Intervals are right-closed
        and include the lowest edge; rows outside explicit edges are excluded and counted as ``outside_bins``.
        """
        from .ae import summarize_ae
        return summarize_ae(self, x_bins=x_bins, z_bins=z_bins)

    def correction(self, *, x_knots: Sequence[float], z_knots: Sequence[float],
                   x_reference: float | None = None, z_reference: float | None = None,
                   loss: Literal["poisson", "mse"] = "poisson", ridge: float = 0.01,
                   max_iter: int = 500, tol: float = 1e-14) -> TwoWayCorrection:
        """Fit a multiplier surface D(x, z) on the prediction; see ``TwoWayCorrection``.

        ``mu = pred * D``, with D bilinear between the knots (in each variable's scale) and constant beyond the
        end knots. A reference, which must be one of that axis's knots, fixes its line of D at 1: with both, D
        holds only the interaction; with neither, D is a smoothed two-way A/E surface relative to pred.
        ``loss`` is ``"poisson"`` (y, pred >= 0) or ``"mse"``. ``ridge`` shrinks free nodes toward 1 and has
        no unit: a node with Fisher information I is pulled toward 1 by about ridge * Ibar / (ridge * Ibar + I),
        where Ibar is the mean information of the free nodes at D = 1. The nodes are fitted by L-BFGS-B
        (``scipy.optimize.minimize``): ``max_iter`` caps its iterations and ``tol`` is its ``ftol``, on the
        penalized loss per unit of weight.
        """
        from .correction import fit_correction
        return fit_correction(self, x_knots=x_knots, z_knots=z_knots, x_reference=x_reference,
                              z_reference=z_reference, loss=loss, ridge=ridge, max_iter=max_iter, tol=tol)
