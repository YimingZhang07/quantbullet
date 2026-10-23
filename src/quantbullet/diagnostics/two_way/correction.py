"""A multiplier surface D(x, z) on a model's predictions, fitted with the predictions held fixed.

D is the tensor product of two tent bases (degree-1 B-splines): bilinear between the knots, in each variable's
scale, and constant beyond the end knots. That is ``TentTransformer(x) ⊗ TentTransformer(z)``. Each row touches
the four corners of its cell, so the normal equations are accumulated with ``np.bincount`` over those corners
and no row-by-node design matrix is built.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .diagnostic import TwoWayDiagnostic

LOSSES = ("poisson", "mse")
_BOUND = 1e-6   # Poisson nodes at or below this are treated as on the positivity bound
SUPPORT = ("rows", "weight", "actual", "expected", "information")
TABLE_VALUES = ("nodes", "se", "lower", "upper", "z_score") + SUPPORT


class _Axis:
    """Knots of one variable in its scale; locates values in their knot cell."""

    def __init__(self, knots, scale: str, name: str):
        knots = np.asarray(knots, dtype=float)
        if knots.ndim != 1 or len(knots) < 2 or not np.isfinite(knots).all() or not (np.diff(knots) > 0).all():
            raise ValueError(f"{name} must contain at least two finite, strictly increasing values")
        if scale == "log" and knots[0] <= 0:
            raise ValueError(f"{name} must be positive on a log scale")
        self.knots, self.scale, self.name = knots, scale, name
        self.t = self.transform(knots)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.log(values) if self.scale == "log" else values

    def reference_index(self, reference, name: str) -> int | None:
        if reference is None:
            return None
        if isinstance(reference, bool) or not np.isscalar(reference) or not np.isfinite(reference):
            raise ValueError(f"{name} must be a finite number or None")
        match = np.flatnonzero(np.isclose(self.knots, float(reference), rtol=1e-12, atol=0))
        if not len(match):
            raise ValueError(f"{name}={reference!r} must be one of the knots {self.knots.tolist()}")
        return int(match[0])

    def locate(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Cell index, position within the cell in [0, 1], and how many values were clamped to the end knots."""
        t = self.transform(values)
        clamped = int(((t < self.t[0]) | (t > self.t[-1])).sum())
        t = np.clip(t, self.t[0], self.t[-1])
        cell = np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2)
        frac = (t - self.t[cell]) / (self.t[cell + 1] - self.t[cell])
        return cell, frac, clamped


def _corners(cx, u, cz, v, n_z: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(flat node index, bilinear weight) for the four corners of each row's cell; nodes are x-major."""
    base = cx * n_z + cz
    return [(base, (1 - u) * (1 - v)), (base + n_z, u * (1 - v)), (base + 1, (1 - u) * v),
            (base + n_z + 1, u * v)]


class _Grid:
    """Sums over rows of corner-weighted quantities, as node vectors and node-by-node matrices."""

    def __init__(self, corners, n_nodes: int):
        self.corners, self.n = corners, n_nodes

    def surface(self, nodes: np.ndarray) -> np.ndarray:
        return sum(weight * nodes[index] for index, weight in self.corners)

    def project(self, row_values: np.ndarray) -> np.ndarray:
        return sum(np.bincount(index, weights=row_values * weight, minlength=self.n)
                   for index, weight in self.corners)

    def diagonal(self, row_weights: np.ndarray) -> np.ndarray:
        return sum(np.bincount(index, weights=row_weights * weight ** 2, minlength=self.n)
                   for index, weight in self.corners)

    def gram(self, row_weights: np.ndarray) -> np.ndarray:
        n = self.n
        total = np.zeros(n * n)
        for a, wa in self.corners:
            for b, wb in self.corners:
                total += np.bincount(a * n + b, weights=row_weights * wa * wb, minlength=n * n)
        return total.reshape(n, n)


@dataclass(frozen=True, eq=False)
class TwoWayCorrection:
    """A fitted multiplier surface D(x, z) on the prediction: mu = pred * D.

    ``nodes``, ``se`` and ``free`` have shape (len(x_knots), len(z_knots)), x down the rows. A reference line
    is fixed at 1 (se 0, ``free`` false). With both references, a node reads as a double ratio: actual over
    model at (x, z), relative to the same ratio at (x, z_reference), and relative again to that ratio on the
    x_reference row. ``se`` is a penalized sandwich SE that treats rows as independent.

    ``support`` spreads each row over its four corners by the interpolation weights: ``rows``, ``weight``,
    ``actual`` = sum(w * y), ``expected`` = sum(w * pred) and the Fisher ``information`` at the fit.
    ``deviance`` holds the weighted mean Poisson deviance (or weighted MSE) at D = 1 (``baseline``) and at the
    fit (``fitted``), and ``change`` = fitted / baseline - 1. ``penalty_scale`` is the Ibar that scales
    ``ridge``. ``clamped`` counts the values beyond the end knots of each axis.
    """

    x: str
    z: str
    x_scale: str
    z_scale: str
    labels: dict
    x_knots: np.ndarray
    z_knots: np.ndarray
    x_reference: float | None
    z_reference: float | None
    loss: str
    ridge: float
    penalty_scale: float
    nodes: np.ndarray
    se: np.ndarray
    free: np.ndarray
    support: dict
    deviance: dict
    n_obs: int
    excluded: dict
    clamped: dict
    iterations: int
    converged: bool

    def label(self, name: str) -> str:
        return self.labels.get(name, name)

    def surface(self, x_values, z_values) -> np.ndarray:
        """D at paired x and z values (broadcast); NaN where a value is nonfinite or not positive on a log axis."""
        xv, zv = np.broadcast_arrays(np.asarray(x_values, dtype=float), np.asarray(z_values, dtype=float))
        x_axis = _Axis(self.x_knots, self.x_scale, "x_knots")
        z_axis = _Axis(self.z_knots, self.z_scale, "z_knots")
        valid = np.isfinite(xv) & np.isfinite(zv)
        if self.x_scale == "log":
            valid &= xv > 0
        if self.z_scale == "log":
            valid &= zv > 0
        out = np.full(xv.shape, np.nan)
        cx, u, _ = x_axis.locate(xv[valid])
        cz, v, _ = z_axis.locate(zv[valid])
        flat = self.nodes.ravel()
        out[valid] = sum(weight * flat[index] for index, weight in _corners(cx, u, cz, v, len(self.z_knots)))
        return out

    def predict(self, df) -> np.ndarray:
        """D for each row of ``df`` (pandas or Polars), read from its x and z columns."""
        from .diagnostic import _float_column
        x = _float_column(df, self.x).to_numpy()
        z = _float_column(df, self.z).to_numpy()
        return self.surface(x, z)

    def table(self, value: str = "nodes", ci: float = .95) -> pd.DataFrame:
        """``value`` at the nodes: x knots down the index, z knots across the columns.

        ``value`` is ``nodes``, ``se``, ``lower`` / ``upper`` (the ``ci`` normal interval), ``z_score`` =
        (node - 1) / se (NaN on reference lines) or a ``support`` key.
        """
        if value not in TABLE_VALUES:
            raise ValueError(f"value must be one of {TABLE_VALUES}")
        if not 0 < ci < 1:
            raise ValueError("ci must be in (0, 1)")
        q = NormalDist().inv_cdf(.5 + ci / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = {
                "nodes": self.nodes, "se": self.se,
                "lower": self.nodes - q * self.se, "upper": self.nodes + q * self.se,
                "z_score": np.where(self.free & (self.se > 0), (self.nodes - 1) / self.se, np.nan),
            }.get(value)
        if values is None:
            values = self.support[value]
        return pd.DataFrame(values, index=pd.Index(self.x_knots, name=self.label(self.x)),
                            columns=pd.Index(self.z_knots, name=self.label(self.z)))

    def plot(self, **kwargs):
        """Heatmap of D above one slice per x knot; see ``quantbullet.plot.draw_two_way_correction``."""
        from quantbullet.plot.two_way import draw_two_way_correction
        return draw_two_way_correction(self, **kwargs)


def _deviance(loss: str, y, mu, w) -> float:
    if loss == "poisson":
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(y > 0, y * np.log(y / mu), 0.) - (y - mu)
        return float(2 * np.sum(w * terms) / np.sum(w))
    return float(np.sum(w * (y - mu) ** 2) / np.sum(w))


def fit_correction(diag: TwoWayDiagnostic, *, x_knots, z_knots, x_reference=None, z_reference=None,
                   loss: str = "poisson", ridge: float = 0.01, max_iter: int = 50,
                   tol: float = 1e-10) -> TwoWayCorrection:
    """Penalized Fisher scoring for the free nodes; see ``TwoWayDiagnostic.correction``."""
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}")
    if isinstance(ridge, bool) or not np.isscalar(ridge) or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be a finite nonnegative number")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isscalar(tol) or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite nonnegative number")
    x_axis = _Axis(x_knots, diag.x_scale, "x_knots")
    z_axis = _Axis(z_knots, diag.z_scale, "z_knots")
    x_ref = x_axis.reference_index(x_reference, "x_reference")
    z_ref = z_axis.reference_index(z_reference, "z_reference")

    y, pred, w = diag._array("y"), diag._array("pred"), diag._array("weight")
    xv, zv = diag._array("x"), diag._array("z")
    excluded = dict(diag.excluded)
    if loss == "poisson":
        if (y < 0).any() or (pred < 0).any():
            raise ValueError("loss='poisson' needs y >= 0 and pred >= 0")
        zero = pred == 0
        if (zero & (y > 0)).any():
            raise ValueError("loss='poisson' cannot fit rows with pred = 0 and y > 0")
        if zero.any():
            excluded["zero_pred"] = int(zero.sum())
            keep = ~zero
            y, pred, w, xv, zv = y[keep], pred[keep], w[keep], xv[keep], zv[keep]
    if not len(y) or not np.sum(w) > 0:
        raise ValueError("no weighted rows to fit")

    n_x, n_z = len(x_axis.knots), len(z_axis.knots)
    cx, u, x_clamped = x_axis.locate(xv)
    cz, v, z_clamped = z_axis.locate(zv)
    grid = _Grid(_corners(cx, u, cz, v, n_z), n_x * n_z)
    del cx, u, cz, v
    free = np.ones((n_x, n_z), dtype=bool)
    if x_ref is not None:
        free[x_ref, :] = False
    if z_ref is not None:
        free[:, z_ref] = False
    free_flat = free.ravel()

    def curvature(D):
        """Fisher weights per row: d2 loss / dD2 contributions."""
        return w * pred / D if loss == "poisson" else w * pred ** 2

    def score(D, mu):
        """Per-row derivative of the loss with respect to D (to be spread over the corners)."""
        return w * (mu - y) / D if loss == "poisson" else w * pred * (mu - y)

    def data_loss(mu):
        if loss == "poisson":
            return float(np.sum(w * (mu - y * np.log(mu))))
        return float(.5 * np.sum(w * (y - mu) ** 2))

    information = grid.diagonal(curvature(np.ones(len(y))))
    penalty_scale = float(information[free_flat].mean())
    if not penalty_scale > 0:
        raise ValueError("the free nodes have no information; check the knots against the data")
    if ridge == 0 and (empty := ~(information > 0) & free_flat).any():
        cells = [(float(x_axis.knots[i // n_z]), float(z_axis.knots[i % n_z])) for i in np.flatnonzero(empty)]
        raise ValueError(f"free nodes without data at (x, z) = {cells}; set ridge > 0 or move the knots")
    strength = ridge * penalty_scale

    def objective(nodes):
        return data_loss(pred * grid.surface(nodes)) + .5 * strength * np.sum((nodes[free_flat] - 1) ** 2)

    nodes = np.ones(n_x * n_z)
    current = objective(nodes)
    eye = np.eye(int(free_flat.sum()))
    converged, iterations = False, 0
    for iterations in range(1, max_iter + 1):
        D = grid.surface(nodes)
        gradient = grid.project(score(D, pred * D))[free_flat] + strength * (nodes[free_flat] - 1)
        hessian = grid.gram(curvature(D))[np.ix_(free_flat, free_flat)] + strength * eye
        # Poisson nodes sitting on the positivity bound that still want to fall stay put for this step;
        # otherwise every step would have to shrink until that node stays positive, stalling the others.
        moving = np.ones(len(gradient), dtype=bool)
        if loss == "poisson":
            moving = ~((nodes[free_flat] <= _BOUND) & (gradient > 0))
        step = np.zeros(len(gradient))
        try:
            step[moving] = np.linalg.solve(hessian[np.ix_(moving, moving)], gradient[moving])
        except np.linalg.LinAlgError as error:
            raise ValueError("singular normal equations; set ridge > 0 or move the knots") from error
        t, accepted = 1., False
        if loss == "poisson" and (falling := step > 0).any():
            # Start from the longest step that keeps every free node positive.
            t = min(1., .99 * float(np.min(nodes[free_flat][falling] / step[falling])))
        while t >= 1e-8:
            trial = nodes.copy()
            trial[free_flat] -= t * step
            if loss != "poisson" or (trial[free_flat] > 0).all():
                candidate = objective(trial)
                if candidate <= current:
                    accepted = True
                    break
            t /= 2
        if not accepted:  # no step lowers the objective: numerically at the optimum
            converged = True
            break
        gain = current - candidate
        nodes, current = trial, candidate
        if gain <= tol * max(abs(current), np.finfo(float).tiny):
            converged = True
            break
    if not converged:
        warnings.warn(f"two-way correction did not converge in {max_iter} iterations", RuntimeWarning, stacklevel=3)

    D = grid.surface(nodes)
    mu = pred * D
    bread = np.linalg.inv(grid.gram(curvature(D))[np.ix_(free_flat, free_flat)] + strength * eye)
    meat = grid.gram(score(D, mu) ** 2)[np.ix_(free_flat, free_flat)]
    se = np.zeros(n_x * n_z)
    se[free_flat] = np.sqrt(np.clip(np.diag(bread @ meat @ bread), 0, None))
    support = {
        "rows": grid.project(np.ones(len(y))), "weight": grid.project(w), "actual": grid.project(w * y),
        "expected": grid.project(w * pred), "information": grid.diagonal(curvature(D)),
    }
    baseline, fitted = _deviance(loss, y, pred, w), _deviance(loss, y, mu, w)
    shape = (n_x, n_z)
    return TwoWayCorrection(
        x=diag.x, z=diag.z, x_scale=diag.x_scale, z_scale=diag.z_scale, labels=dict(diag.labels),
        x_knots=x_axis.knots, z_knots=z_axis.knots,
        x_reference=None if x_ref is None else float(x_axis.knots[x_ref]),
        z_reference=None if z_ref is None else float(z_axis.knots[z_ref]),
        loss=loss, ridge=float(ridge), penalty_scale=penalty_scale,
        nodes=nodes.reshape(shape), se=se.reshape(shape), free=free,
        support={key: value.reshape(shape) for key, value in support.items()},
        deviance={"baseline": baseline, "fitted": fitted,
                  "change": fitted / baseline - 1 if baseline > 0 else np.nan},
        n_obs=len(y), excluded=excluded, clamped={"x": x_clamped, "z": z_clamped},
        iterations=iterations, converged=converged)
