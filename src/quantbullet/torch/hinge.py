from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_col(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1, 1) if x.ndim == 1 else x


@dataclass
class HingeExport:
    """Serialisable snapshot of a trained Hinge."""
    shape: str
    b: float
    a: float
    t: np.ndarray
    c: np.ndarray
    x_min: float
    x_max: float


class Hinge(nn.Module):
    """
    Piecewise-linear hinge function with built-in input scaling.

        f(x) = b + a * z + sum_k  c_k * relu(z - t_k)

    where z = clip-and-scale(x) maps the raw input to [0, 1] internally.

    Parameters
    ----------
    n_knots : int
        Number of evenly-spaced knots (ignored when *knots* is given).
    knots : array-like, optional
        Explicit knot positions **in the original feature space**.
        When provided, *n_knots* and *x_range* are ignored; the input
        range is inferred from ``knots[0]`` / ``knots[-1]``.
    x_range : (float, float), optional
        ``(min, max)`` of the feature domain.  Required when using
        *n_knots* (ignored when *knots* is given).
    shape : {"generic", "concave", "convex"}
        Constraint on the slope-change coefficients *c_k*:

        * ``"generic"``  -- unconstrained
        * ``"concave"``  -- c_k <= 0  (slope can only decrease)
        * ``"convex"``   -- c_k >= 0  (slope can only increase)
    monotone : {None, "increasing", "decreasing"}
        Optional monotonicity constraint on the overall slope.
    intercept : bool
        If *True* (default), ``b`` is a learnable parameter.
        If *False*, ``b`` is fixed at zero — the hinge learns shape
        only and is zero-centred.  Use this in additive models where
        a single global intercept captures the level.
    clip : bool
        If *True* (default), clip the raw input to *x_range* before
        scaling.  Keeps predictions stable outside the training domain.

    Examples
    --------
    >>> h = Hinge(n_knots=20, x_range=(1.0, 1.05),
    ...           shape="concave", monotone="increasing")
    >>> y = h(x_mvoc)                       # pass raw MVOC values
    >>> loss = h.smoothness_penalty()        # regularisation term
    """

    _SHAPES = ("generic", "concave", "convex")
    _MONOTONES = (None, "increasing", "decreasing")

    def __init__(
        self,
        n_knots: int = 10,
        knots: Optional[np.ndarray] = None,
        x_range: Optional[tuple[float, float]] = None,
        shape: str = "generic",
        monotone: Optional[str] = None,
        intercept: bool = True,
        clip: bool = True,
    ):
        super().__init__()

        # --- validate shape / monotone --------------------------------
        if shape not in self._SHAPES:
            raise ValueError(f"shape must be one of {self._SHAPES}, got {shape!r}")
        if monotone not in self._MONOTONES:
            raise ValueError(f"monotone must be one of {self._MONOTONES}, got {monotone!r}")
        self._shape = shape
        self._monotone = monotone
        self._clip = clip

        # --- resolve knots & x_range ----------------------------------
        if knots is not None:
            knots = np.asarray(knots, dtype=np.float64)
            if knots.ndim != 1 or len(knots) < 2:
                raise ValueError("knots must be a 1-D array with >= 2 elements")
            if not np.all(np.diff(knots) > 0):
                raise ValueError("knots must be strictly increasing")
            x_lo, x_hi = float(knots[0]), float(knots[-1])
            knots_01 = (knots - x_lo) / (x_hi - x_lo)
        else:
            if x_range is None:
                raise ValueError("x_range is required when knots is not given")
            x_lo, x_hi = float(x_range[0]), float(x_range[1])
            if x_lo >= x_hi:
                raise ValueError("x_range[0] must be < x_range[1]")
            knots_01 = np.linspace(0.0, 1.0, n_knots)

        self.register_buffer("x_min", torch.tensor(x_lo, dtype=torch.float32))
        self.register_buffer("x_max", torch.tensor(x_hi, dtype=torch.float32))

        knots_t = torch.as_tensor(knots_01, dtype=torch.float32)
        self.K = len(knots_t)
        self.register_buffer("t_fixed", knots_t)

        # --- learnable parameters -------------------------------------
        if intercept:
            self.b = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("b", torch.tensor(0.0))
        self.a_raw = nn.Parameter(torch.tensor(0.0))
        self.c_raw = nn.Parameter(torch.zeros(self.K))

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------
    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_col(x)
        if self._clip:
            x = x.clamp(self.x_min, self.x_max)
        return ((x - self.x_min) / (self.x_max - self.x_min).clamp(min=1e-12)).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Constrained parameters
    # ------------------------------------------------------------------
    def _effective_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        shape, mono = self._shape, self._monotone

        # --- slope-change coefficients c ------------------------------
        if shape == "concave":
            c = -F.softplus(self.c_raw)
        elif shape == "convex":
            c = F.softplus(self.c_raw)
        else:
            c = self.c_raw

        # --- initial slope a ------------------------------------------
        if shape == "generic":
            a = self._generic_monotone(c)
        elif shape == "concave":
            a = self._concave_monotone(c)
        elif shape == "convex":
            a = self._convex_monotone(c)
        return a, c

    def _generic_monotone(self, c: torch.Tensor) -> torch.Tensor:
        if self._monotone is None:
            return self.a_raw
        slopes = torch.cat([self.a_raw.view(1), self.a_raw + torch.cumsum(c, 0)])
        zero = slopes.new_zeros(())
        if self._monotone == "increasing":
            return self.a_raw - torch.minimum(slopes.min(), zero)
        return self.a_raw - torch.maximum(slopes.max(), zero)

    def _concave_monotone(self, c: torch.Tensor) -> torch.Tensor:
        if self._monotone == "increasing":
            return F.softplus(self.a_raw) + F.softplus(self.c_raw).sum() + 1e-3
        if self._monotone == "decreasing":
            return -F.softplus(self.a_raw) - 1e-3
        return self.a_raw

    def _convex_monotone(self, c: torch.Tensor) -> torch.Tensor:
        if self._monotone == "increasing":
            return F.softplus(self.a_raw)
        if self._monotone == "decreasing":
            return -F.softplus(self.a_raw) - c.sum() - 1e-3
        return self.a_raw

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._scale(x)
        t = self.t_fixed.view(1, -1)
        a, c = self._effective_params()
        relu_terms = torch.relu(z - t)
        return self.b + a * z + (relu_terms * c.view(1, -1)).sum(dim=1, keepdim=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def knots(self, original: bool = True) -> torch.Tensor:
        """Return knot positions.  *original=True* maps back to feature space."""
        t01 = self.t_fixed
        if original:
            return self.x_min + t01 * (self.x_max - self.x_min)
        return t01

    def smoothness_penalty(
        self,
        kind: Literal["c_l2", "c_diff"] = "c_diff",
        reduction: Literal["mean", "sum"] = "mean",
    ) -> torch.Tensor:
        _, c = self._effective_params()
        if kind == "c_l2":
            vals = c.pow(2)
        elif kind == "c_diff":
            if c.numel() < 2:
                return c.new_zeros(())
            vals = (c[1:] - c[:-1]).pow(2)
        else:
            raise ValueError("kind must be 'c_l2' or 'c_diff'")
        return vals.mean() if reduction == "mean" else vals.sum()

    @torch.no_grad()
    def export_params(self) -> HingeExport:
        a, c = self._effective_params()
        return HingeExport(
            shape=self._shape,
            b=float(self.b.cpu()),
            a=float(a.cpu()),
            t=self.t_fixed.cpu().numpy().astype(np.float32),
            c=c.cpu().numpy().astype(np.float32),
            x_min=float(self.x_min.cpu()),
            x_max=float(self.x_max.cpu()),
        )

    def plot(self, n_points: int = 200, ax=None, show_knots: bool = True):
        """Plot the hinge curve.  See ``quantbullet.torch.plot.plot_hinge``."""
        from .plot import plot_hinge
        return plot_hinge(self, n_points=n_points, ax=ax, show_knots=show_knots)

    def __repr__(self) -> str:
        mono = f", monotone={self._monotone!r}" if self._monotone else ""
        x_lo = float(self.x_min.detach())
        x_hi = float(self.x_max.detach())
        return (
            f"Hinge(K={self.K}, shape={self._shape!r}{mono}, "
            f"x_range=({x_lo:.4g}, {x_hi:.4g}))"
        )
