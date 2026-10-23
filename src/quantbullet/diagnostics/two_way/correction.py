"""A multiplier surface D(x, z) on a model's predictions, fitted with the predictions held fixed.

D is the tensor product of two degree-1 B-spline (tent) bases: bilinear between the knots, in each variable's
scale, and constant beyond the end knots (the same basis as ``TentTransformer``). Each axis's basis comes from
``scipy.interpolate.BSpline.design_matrix``; their row-wise product is a sparse design ``Z`` with four entries
per row. Because each row of ``Z`` sums to 1, D = 1 + Z[:, free] @ delta: the prediction is a fixed offset, a
reference line is simply left out of ``free``, and the ridge pulls delta toward 0. ``scipy.optimize.minimize``
(L-BFGS-B) fits delta; for Poisson, bounds keep every node positive.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.interpolate import BSpline
from scipy.optimize import minimize

if TYPE_CHECKING:
    from .diagnostic import TwoWayDiagnostic

LOSSES = ("poisson", "mse")
SUPPORT = ("rows", "weight", "actual", "expected", "information")
TABLE_VALUES = ("nodes", "se", "lower", "upper", "z_score") + SUPPORT
_FLOOR = 1e-10   # Poisson nodes stay at or above this, so mu stays positive


class _Axis:
    """Knots of one variable in its scale, and the tent basis of values on them."""

    def __init__(self, knots, scale: str, name: str):
        knots = np.asarray(knots, dtype=float)
        if knots.ndim != 1 or len(knots) < 2 or not np.isfinite(knots).all() or not (np.diff(knots) > 0).all():
            raise ValueError(f"{name} must contain at least two finite, strictly increasing values")
        if scale == "log" and knots[0] <= 0:
            raise ValueError(f"{name} must be positive on a log scale")
        self.knots, self.scale = knots, scale
        self.t = self.transform(knots)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.log(values) if self.scale == "log" else np.asarray(values, dtype=float)

    def reference_index(self, reference, name: str) -> int | None:
        if reference is None:
            return None
        if isinstance(reference, bool) or not np.isscalar(reference) or not np.isfinite(reference):
            raise ValueError(f"{name} must be a finite number or None")
        match = np.flatnonzero(np.isclose(self.knots, float(reference), rtol=1e-12, atol=0))
        if not len(match):
            raise ValueError(f"{name}={reference!r} must be one of the knots {self.knots.tolist()}")
        return int(match[0])

    def basis(self, values: np.ndarray) -> tuple[sp.csr_array, int]:
        """Tent basis (rows, knots), two entries per row, and the number of values clamped to the end knots."""
        t = self.transform(values)
        clamped = int(((t < self.t[0]) | (t > self.t[-1])).sum())
        knots = np.r_[self.t[0], self.t, self.t[-1]]   # degree 1: one repeated knot at each end
        return BSpline.design_matrix(np.clip(t, self.t[0], self.t[-1]), knots, 1), clamped


def _design(x_basis: sp.csr_array, z_basis: sp.csr_array) -> sp.csr_array:
    """Row-wise tensor product of two tent bases: four entries per row, nodes in x-major order."""
    n, n_z = x_basis.shape[0], z_basis.shape[1]
    xi, xv = x_basis.indices.reshape(n, 2), x_basis.data.reshape(n, 2)
    zi, zv = z_basis.indices.reshape(n, 2), z_basis.data.reshape(n, 2)
    indices = (xi[:, :, None] * n_z + zi[:, None, :]).ravel()
    data = (xv[:, :, None] * zv[:, None, :]).ravel()
    return sp.csr_array((data, indices, np.arange(0, 4 * n + 1, 4)), shape=(n, x_basis.shape[1] * n_z))


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
        valid = np.isfinite(xv) & np.isfinite(zv)
        if self.x_scale == "log":
            valid &= xv > 0
        if self.z_scale == "log":
            valid &= zv > 0
        out = np.full(xv.shape, np.nan)
        if valid.any():
            x_basis, _ = _Axis(self.x_knots, self.x_scale, "x_knots").basis(xv[valid])
            z_basis, _ = _Axis(self.z_knots, self.z_scale, "z_knots").basis(zv[valid])
            out[valid] = _design(x_basis, z_basis) @ self.nodes.ravel()
        return out

    def predict(self, df) -> np.ndarray:
        """D for each row of ``df`` (pandas or Polars), read from its x and z columns."""
        from .diagnostic import _float_column
        return self.surface(_float_column(df, self.x).to_numpy(), _float_column(df, self.z).to_numpy())

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


def fit_correction(diag: TwoWayDiagnostic, *, x_knots, z_knots, x_reference=None, z_reference=None,
                   loss: str = "poisson", ridge: float = 0.01, max_iter: int = 500,
                   tol: float = 1e-14) -> TwoWayCorrection:
    """L-BFGS-B on the free nodes with the prediction fixed; see ``TwoWayDiagnostic.correction``."""
    from sklearn.metrics import mean_poisson_deviance, mean_squared_error

    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}")
    if isinstance(ridge, bool) or not np.isscalar(ridge) or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be a finite nonnegative number")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isscalar(tol) or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be a finite nonnegative number")
    poisson = loss == "poisson"
    x_axis = _Axis(x_knots, diag.x_scale, "x_knots")
    z_axis = _Axis(z_knots, diag.z_scale, "z_knots")
    x_ref = x_axis.reference_index(x_reference, "x_reference")
    z_ref = z_axis.reference_index(z_reference, "z_reference")

    y, pred, w = diag._array("y"), diag._array("pred"), diag._array("weight")
    xv, zv = diag._array("x"), diag._array("z")
    excluded = dict(diag.excluded)
    if poisson:
        if (y < 0).any() or (pred < 0).any():
            raise ValueError("loss='poisson' needs y >= 0 and pred >= 0")
        zero = pred == 0
        if (zero & (y > 0)).any():
            raise ValueError("loss='poisson' cannot fit rows with pred = 0 and y > 0")
        if zero.any():
            excluded["zero_pred"] = int(zero.sum())
            keep = ~zero
            y, pred, w, xv, zv = y[keep], pred[keep], w[keep], xv[keep], zv[keep]
    total_weight = float(np.sum(w))
    if not len(y) or not total_weight > 0:
        raise ValueError("no weighted rows to fit")

    x_basis, x_clamped = x_axis.basis(xv)
    z_basis, z_clamped = z_axis.basis(zv)
    Z = _design(x_basis, z_basis)
    del x_basis, z_basis
    shape = (len(x_axis.knots), len(z_axis.knots))
    free = np.ones(shape, dtype=bool)
    if x_ref is not None:
        free[x_ref, :] = False
    if z_ref is not None:
        free[:, z_ref] = False
    free_index = np.flatnonzero(free)
    A = sp.csr_array(Z[:, free_index].multiply(pred[:, None]))   # mu = pred + A @ delta

    # Fisher information of each free node at D = 1 sets the unit of the ridge.
    information = A.multiply(A).T @ (w / pred if poisson else w)
    penalty_scale = float(information.mean())
    if not penalty_scale > 0:
        raise ValueError("the free nodes have no information; check the knots against the data")
    if ridge == 0 and (empty := ~(information > 0)).any():
        cells = [(float(x_axis.knots[i // shape[1]]), float(z_axis.knots[i % shape[1]]))
                 for i in free_index[empty]]
        raise ValueError(f"free nodes without data at (x, z) = {cells}; set ridge > 0 or move the knots")
    strength = ridge * penalty_scale

    # L-BFGS-B works on v = scale * delta: each node in units of its own information at D = 1 (plus the
    # ridge), so the Hessian is near the identity and it needs about half the iterations.
    scale = np.sqrt((information + strength) / penalty_scale)
    wy, w_gap = w * y, w * (pred - y)

    def objective(v):
        """Penalized loss minus its value at D = 1, in units of the mean node information, with its gradient.

        Summing each row's change (rather than differencing two large sums) keeps the value precise near the
        optimum, and the units give a curvature near 1 for either loss, so L-BFGS-B's tolerances fit both.
        """
        delta = v / scale
        change = A @ delta                                   # mu - pred
        if poisson:
            value = w @ change - wy @ np.log1p(change / pred)
            d_mu = w - wy / (pred + change)
        else:
            value = .5 * (w @ change ** 2) + w_gap @ change
            d_mu = w * change + w_gap
        value += .5 * strength * delta @ delta
        return value / penalty_scale, (A.T @ d_mu + strength * delta) / (penalty_scale * scale)

    bounds = [((_FLOOR - 1) * s, None) for s in scale] if poisson else None
    result = minimize(objective, np.zeros(len(free_index)), jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": max_iter, "ftol": tol, "gtol": 1e-10})
    if not result.success:
        warnings.warn(f"two-way correction did not converge: {result.message}", RuntimeWarning, stacklevel=3)

    delta = result.x / scale
    mu = pred + A @ delta
    curvature = w / mu if poisson else w            # Fisher weight per row on mu
    score = w * (y - mu) / mu if poisson else w * (y - mu)
    hessian = (A.T @ A.multiply(curvature[:, None])).toarray() + strength * np.eye(len(free_index))
    meat = (A.T @ A.multiply((score ** 2)[:, None])).toarray()
    try:
        bread = np.linalg.inv(hessian)
    except np.linalg.LinAlgError as error:
        raise ValueError("singular information matrix; set ridge > 0 or move the knots") from error
    nodes, se = np.ones(Z.shape[1]), np.zeros(Z.shape[1])
    nodes[free_index] = 1 + delta
    se[free_index] = np.sqrt(np.clip(np.diag(bread @ meat @ bread), 0, None))

    sums = {"rows": np.ones(len(y)), "weight": w, "actual": w * y, "expected": w * pred}
    support = {key: Z.T @ values for key, values in sums.items()}
    support["information"] = Z.multiply(Z).T @ (curvature * pred ** 2)
    deviance = mean_poisson_deviance if poisson else mean_squared_error
    baseline, fitted = deviance(y, pred, sample_weight=w), deviance(y, mu, sample_weight=w)
    return TwoWayCorrection(
        x=diag.x, z=diag.z, x_scale=diag.x_scale, z_scale=diag.z_scale, labels=dict(diag.labels),
        x_knots=x_axis.knots, z_knots=z_axis.knots,
        x_reference=None if x_ref is None else float(x_axis.knots[x_ref]),
        z_reference=None if z_ref is None else float(z_axis.knots[z_ref]),
        loss=loss, ridge=float(ridge), penalty_scale=penalty_scale,
        nodes=nodes.reshape(shape), se=se.reshape(shape), free=free,
        support={key: np.asarray(value, dtype=float).reshape(shape) for key, value in support.items()},
        deviance={"baseline": float(baseline), "fitted": float(fitted),
                  "change": float(fitted / baseline - 1) if baseline > 0 else np.nan},
        n_obs=len(y), excluded=excluded, clamped={"x": x_clamped, "z": z_clamped},
        iterations=int(result.nit), converged=bool(result.success))
