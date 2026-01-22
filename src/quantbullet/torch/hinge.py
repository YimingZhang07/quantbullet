# hinge.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_col(x: torch.Tensor) -> torch.Tensor:
    """Convert a 1D or 2D tensor to a 2D column vector."""
    return x.view(-1, 1) if x.ndim == 1 else x


class _PlotMixin:
    """Mixin providing plot functionality for all hinge models."""
    
    def plot(self, n_points: int = 200, ax=None, show_knots: bool = True):
        """Plot the hinge function. See quantbullet.torch.plot.plot_hinge for details."""
        from .plot import plot_hinge
        return plot_hinge(self, n_points=n_points, ax=ax, show_knots=show_knots)


@dataclass
class HingeExport:
    kind: str
    b: float
    a: float
    t: np.ndarray          # (K,)
    c: np.ndarray          # (K,) slope-change coeffs, generic/concave/convex
    # for flat-tail concave
    m: Optional[float] = None
    w: Optional[np.ndarray] = None


class GenericHinge(_PlotMixin, nn.Module):
    """
    f(x)=b + a*x + sum_k c_k * relu(x - t_k)

    This is the MOST generic piecewise linear hinge model.
    - knots t_k can be fixed (provided) or learnable and sorted in [0,1].
    - c_k are free (no concavity/convexity constraints here).
    
    Args:
        K: Number of knots (required only if fixed_knots is not provided)
        fixed_knots: Optional array of fixed knot positions in [0,1]. If provided,
                     K is inferred from length. Should be sorted and in [0,1].
    """
    def __init__(self, K: Optional[int] = None, fixed_knots: Optional[np.ndarray | torch.Tensor] = None):
        super().__init__()
        
        if fixed_knots is not None:
            # Register as buffer (non-trainable)
            # check the knots are sorted and in [0,1]
            if not np.all(np.diff(fixed_knots) > 0):
                raise ValueError("fixed_knots must be sorted and in [0,1]")
            if not np.all(fixed_knots >= 0) or not np.all(fixed_knots <= 1):
                raise ValueError("fixed_knots must be in [0,1]")
            knots_tensor = torch.as_tensor(fixed_knots, dtype=torch.float32)
            self.K = knots_tensor.shape[0]
            self.register_buffer('t_fixed', knots_tensor)
            self._fixed_knots = True
        else:
            if K is None:
                raise ValueError("K must be provided when fixed_knots is not given")
            self.K = K
            self.t_incr_raw = nn.Parameter(torch.zeros(K)) # for sorted knots
            self._fixed_knots = False
        
        self.b = nn.Parameter(torch.tensor(0.0))
        self.a = nn.Parameter(torch.tensor(0.0))
        self.c = nn.Parameter(torch.zeros(self.K))          # unconstrained

    def knots(self) -> torch.Tensor:
        if self._fixed_knots:
            return self.t_fixed
        # t_incr_raw can store any real number, in this function we convert it to a sorted list of knots in [0,1]
        # this is done by making them positive and then cumsumming them ( so they are all positive and increasing )
        # then we normalize them to (0,1] by dividing by the last value
        # finally we clamp them to [0,1]
        incr = F.softplus(self.t_incr_raw) + 1e-6
        t = torch.cumsum(incr, dim=0)
        t = t / (t[-1] + 1e-12)        # normalize to (0,1]
        return t.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_col(x) # convert to column vector， this is (N,1)
        t = self.knots().view(1, -1)   # (1,K)
        # x - t 利用广播从 (N,1) 和 (1,K) 得到 (N,K)：
        # 第 n 行第 k 列是 x_n - t_k
        # 再 ReLU：变成 max(0,x_n - t_k)
        relu_terms = torch.relu(x - t) # (N,K)

        # relu_terms * self.c.view(1, -1) 得到 (N,K)，每个样本的每个knot对应一个c_k
        # 然后我们sum(dim=1, keepdim=True)，得到 (N,1)，即每个样本的hinge值
        y = self.b + self.a * x + (relu_terms * self.c.view(1, -1)).sum(dim=1, keepdim=True)
        return y

    @torch.no_grad()
    def export_params(self) -> HingeExport:
        t = self.knots().detach().cpu().numpy().astype(np.float32)
        c = self.c.detach().cpu().numpy().astype(np.float32)
        return HingeExport(
            kind="generic",
            b=float(self.b.detach().cpu().item()),
            a=float(self.a.detach().cpu().item()),
            t=t,
            c=c,
        )


class ConcaveHinge(_PlotMixin, nn.Module):
    """
    Concave version of GenericHinge:
      enforce slope-change c_k <= 0  via c_k = -softplus(u_k)
    Optionally enforce monotonicity via softplus transformations.

    f(x)=b + a*x + sum_k c_k relu(x-t_k), with c_k <= 0 => slope decreases => concave.
    
    Args:
        K: Number of knots (required only if fixed_knots is not provided)
        monotone_increasing: If True, enforce slope >= 0 everywhere
        monotone_decreasing: If True, enforce slope <= 0 everywhere
        fixed_knots: Optional array of fixed knot positions in [0,1]. If provided,
                     K is inferred from length. Should be sorted and in [0,1].
    """
    def __init__(
        self, 
        K: Optional[int] = None, 
        monotone_increasing: bool = False,
        monotone_decreasing: bool = False,
        fixed_knots: Optional[np.ndarray | torch.Tensor] = None
    ):
        super().__init__()
        
        if monotone_increasing and monotone_decreasing:
            raise ValueError("Cannot be both monotone increasing and decreasing")
        
        self.monotone_increasing = monotone_increasing
        self.monotone_decreasing = monotone_decreasing
        
        # Handle knots (same pattern as GenericHinge)
        if fixed_knots is not None:
            if not np.all(np.diff(fixed_knots) > 0):
                raise ValueError("fixed_knots must be sorted and in [0,1]")
            if not np.all(fixed_knots >= 0) or not np.all(fixed_knots <= 1):
                raise ValueError("fixed_knots must be in [0,1]")
            knots_tensor = torch.as_tensor(fixed_knots, dtype=torch.float32)
            self.K = knots_tensor.shape[0]
            self.register_buffer('t_fixed', knots_tensor)
            self._fixed_knots = True
        else:
            if K is None:
                raise ValueError("K must be provided when fixed_knots is not given")
            self.K = K
            self.t_incr_raw = nn.Parameter(torch.zeros(K))
            self._fixed_knots = False
        
        # Parameters
        self.b = nn.Parameter(torch.tensor(0.0))
        self.a_raw = nn.Parameter(torch.tensor(0.0))
        self.c_raw = nn.Parameter(torch.zeros(self.K))  # -> negative

    def knots(self) -> torch.Tensor:
        if self._fixed_knots:
            return self.t_fixed
        incr = F.softplus(self.t_incr_raw) + 1e-6
        t = torch.cumsum(incr, dim=0)
        t = t / (t[-1] + 1e-12)
        return t.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_col(x)
        t = self.knots().view(1, -1)
        
        c_raw_softplus = F.softplus(self.c_raw)
        c = -c_raw_softplus  # Always negative for concave
        
        if self.monotone_increasing:
            # Enforce slope >= 0: a + sum(c_k) >= 0
            # Since c_k < 0, need a >= sum|c_k|
            c_sum = c_raw_softplus.sum()
            a = F.softplus(self.a_raw) + c_sum + 1e-3
            
        elif self.monotone_decreasing:
            # Enforce slope <= 0: a + sum(c_k) <= 0
            # Since c_k < 0, this is already easier to satisfy
            # Just force a <= 0
            a = -F.softplus(self.a_raw) - 1e-3
            
        else:
            # No monotonicity constraint
            a = self.a_raw
        
        relu_terms = torch.relu(x - t)
        y = self.b + a * x + (relu_terms * c.view(1, -1)).sum(dim=1, keepdim=True)
        return y

    @torch.no_grad()
    def export_params(self) -> HingeExport:
        t = self.knots().detach().cpu().numpy().astype(np.float32)
        
        c_raw_softplus = F.softplus(self.c_raw)
        c = (-c_raw_softplus).detach().cpu().numpy().astype(np.float32)
        
        if self.monotone_increasing:
            c_sum = c_raw_softplus.sum()
            a = F.softplus(self.a_raw) + c_sum + 1e-3
        elif self.monotone_decreasing:
            a = -F.softplus(self.a_raw) - 1e-3
        else:
            a = self.a_raw
        
        return HingeExport(
            kind="concave",
            b=float(self.b.detach().cpu().item()),
            a=float(a.detach().cpu().item()),
            t=t,
            c=c,
        )


class ConvexHinge(_PlotMixin, nn.Module):
    """
    Convex version of GenericHinge:
      enforce slope-change c_k >= 0 via c_k = softplus(u_k)
    Optionally enforce monotonicity.

    f(x)=b + a*x + sum_k c_k relu(x-t_k), with c_k >= 0 => slope increases => convex.
    
    Args:
        K: Number of knots (required only if fixed_knots is not provided)
        monotone_increasing: If True, enforce slope >= 0 everywhere
        monotone_decreasing: If True, enforce slope <= 0 everywhere
        fixed_knots: Optional array of fixed knot positions in [0,1]. If provided,
                     K is inferred from length. Should be sorted and in [0,1].
    """
    def __init__(
        self, 
        K: Optional[int] = None, 
        monotone_increasing: bool = False,
        monotone_decreasing: bool = False,
        fixed_knots: Optional[np.ndarray | torch.Tensor] = None
    ):
        super().__init__()
        
        if monotone_increasing and monotone_decreasing:
            raise ValueError("Cannot be both monotone increasing and decreasing")
        
        self.monotone_increasing = monotone_increasing
        self.monotone_decreasing = monotone_decreasing
        
        # Handle knots (same pattern as ConcaveHinge)
        if fixed_knots is not None:
            if not np.all(np.diff(fixed_knots) > 0):
                raise ValueError("fixed_knots must be sorted and in [0,1]")
            if not np.all(fixed_knots >= 0) or not np.all(fixed_knots <= 1):
                raise ValueError("fixed_knots must be in [0,1]")
            knots_tensor = torch.as_tensor(fixed_knots, dtype=torch.float32)
            self.K = knots_tensor.shape[0]
            self.register_buffer('t_fixed', knots_tensor)
            self._fixed_knots = True
        else:
            if K is None:
                raise ValueError("K must be provided when fixed_knots is not given")
            self.K = K
            self.t_incr_raw = nn.Parameter(torch.zeros(K))
            self._fixed_knots = False
        
        # Parameters
        self.b = nn.Parameter(torch.tensor(0.0))
        self.a_raw = nn.Parameter(torch.tensor(0.0))
        self.c_raw = nn.Parameter(torch.zeros(self.K))  # -> positive

    def knots(self) -> torch.Tensor:
        if self._fixed_knots:
            return self.t_fixed
        incr = F.softplus(self.t_incr_raw) + 1e-6
        t = torch.cumsum(incr, dim=0)
        t = t / (t[-1] + 1e-12)
        return t.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_col(x)
        t = self.knots().view(1, -1)
        
        c = F.softplus(self.c_raw)  # >= 0 for convex
        
        if self.monotone_increasing:
            # Slope = a + sum(c_k) >= 0
            # Since c_k >= 0, just need a >= 0
            a = F.softplus(self.a_raw)
            
        elif self.monotone_decreasing:
            # Slope = a + sum(c_k) <= 0
            # Since c_k >= 0, need a <= -sum(c_k)
            c_sum = c.sum()
            a = -F.softplus(self.a_raw) - c_sum - 1e-3
            
        else:
            # No monotonicity constraint
            a = self.a_raw
        
        relu_terms = torch.relu(x - t)
        y = self.b + a * x + (relu_terms * c.view(1, -1)).sum(dim=1, keepdim=True)
        return y

    @torch.no_grad()
    def export_params(self) -> HingeExport:
        t = self.knots().detach().cpu().numpy().astype(np.float32)
        
        c_raw_softplus = F.softplus(self.c_raw)
        c = c_raw_softplus.detach().cpu().numpy().astype(np.float32)
        
        if self.monotone_increasing:
            a = F.softplus(self.a_raw)
        elif self.monotone_decreasing:
            c_sum = c_raw_softplus.sum()
            a = -F.softplus(self.a_raw) - c_sum - 1e-3
        else:
            a = self.a_raw
        
        return HingeExport(
            kind="convex",
            b=float(self.b.detach().cpu().item()),
            a=float(a.detach().cpu().item()),
            t=t,
            c=c,
        )


class _MinMaxScalerMixin:
    """Mixin providing min-max scaling and clipping functionality."""
    
    def _init_scaler(
        self,
        x_min: float | None = None,
        x_max: float | None = None,
        clip_lo: float | None = None,
        clip_hi: float | None = None,
    ):
        """Initialize scaling buffers. Call this from __init__."""
        if x_min is None and clip_lo is not None:
            x_min = clip_lo
        if x_max is None and clip_hi is not None:
            x_max = clip_hi
        if x_min is None or x_max is None:
            raise ValueError("x_min and x_max must be provided or inferred from clip_lo/clip_hi.")

        self.register_buffer("x_min", torch.tensor(float(x_min), dtype=torch.float32))
        self.register_buffer("x_max", torch.tensor(float(x_max), dtype=torch.float32))
        self.register_buffer("clip_lo", torch.tensor(float(clip_lo), dtype=torch.float32) if clip_lo is not None else float("nan"))
        self.register_buffer("clip_hi", torch.tensor(float(clip_hi), dtype=torch.float32) if clip_hi is not None else float("nan"))

    def _clip_raw(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isfinite(self.clip_lo):
            x = torch.max(x, self.clip_lo)
        if torch.isfinite(self.clip_hi):
            x = torch.min(x, self.clip_hi)
        return x

    def scale(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Scale raw input to [0,1] with optional clipping."""
        x = x_raw.view(-1, 1) if x_raw.ndim == 1 else x_raw
        x = self._clip_raw(x)
        denom = (self.x_max - self.x_min).clamp(min=1e-12)
        x01 = (x - self.x_min) / denom
        return x01.clamp(0.0, 1.0)
    
    def _export_scaler_params(self) -> dict:
        """Export scaler parameters for serialization."""
        return {
            "x_min": float(self.x_min.cpu().item()),
            "x_max": float(self.x_max.cpu().item()),
            "clip_lo": None if not torch.isfinite(self.clip_lo) else float(self.clip_lo.cpu().item()),
            "clip_hi": None if not torch.isfinite(self.clip_hi) else float(self.clip_hi.cpu().item()),
        }


class MinMaxScaledGenericHinge(_PlotMixin, _MinMaxScalerMixin, nn.Module):
    """GenericHinge with automatic min-max scaling and optional clipping."""
    
    def __init__(
        self,
        K: Optional[int] = None,
        x_min: float | None = None,
        x_max: float | None = None,
        clip_lo: float | None = None,
        clip_hi: float | None = None,
        fixed_knots: Optional[np.ndarray | torch.Tensor] = None,
    ):
        super().__init__()
        self.hinge = GenericHinge(K=K, fixed_knots=fixed_knots)
        self._init_scaler(x_min=x_min, x_max=x_max, clip_lo=clip_lo, clip_hi=clip_hi)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        x01 = self.scale(x_raw)
        return self.hinge(x01)

    @torch.no_grad()
    def export_params(self):
        exp = self.hinge.export_params()
        return {
            "hinge": {
                "kind": exp.kind,
                "b": exp.b,
                "a": exp.a,
                "t": exp.t.tolist(),
                "c": exp.c.tolist(),
            },
            "scaler": self._export_scaler_params(),
        }


class MinMaxScaledConcaveHinge(_PlotMixin, _MinMaxScalerMixin, nn.Module):
    """ConcaveHinge with automatic min-max scaling and optional clipping."""
    
    def __init__(
        self,
        K: Optional[int] = None,
        monotone_increasing: bool = False,
        monotone_decreasing: bool = False,
        x_min: float | None = None,
        x_max: float | None = None,
        clip_lo: float | None = None,
        clip_hi: float | None = None,
        fixed_knots: Optional[np.ndarray | torch.Tensor] = None,
    ):
        super().__init__()
        self.hinge = ConcaveHinge(
            K=K, 
            monotone_increasing=monotone_increasing,
            monotone_decreasing=monotone_decreasing,
            fixed_knots=fixed_knots
        )
        self._init_scaler(x_min=x_min, x_max=x_max, clip_lo=clip_lo, clip_hi=clip_hi)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        x01 = self.scale(x_raw)
        return self.hinge(x01)

    @torch.no_grad()
    def export_params(self):
        exp = self.hinge.export_params()
        return {
            "hinge": {
                "kind": exp.kind,
                "b": exp.b,
                "a": exp.a,
                "t": exp.t.tolist(),
                "c": exp.c.tolist(),
            },
            "scaler": self._export_scaler_params(),
        }


class MinMaxScaledConvexHinge(_PlotMixin, _MinMaxScalerMixin, nn.Module):
    """ConvexHinge with automatic min-max scaling and optional clipping."""
    
    def __init__(
        self,
        K: Optional[int] = None,
        monotone_increasing: bool = False,
        monotone_decreasing: bool = False,
        x_min: float | None = None,
        x_max: float | None = None,
        clip_lo: float | None = None,
        clip_hi: float | None = None,
        fixed_knots: Optional[np.ndarray | torch.Tensor] = None,
    ):
        super().__init__()
        self.hinge = ConvexHinge(
            K=K, 
            monotone_increasing=monotone_increasing,
            monotone_decreasing=monotone_decreasing,
            fixed_knots=fixed_knots
        )
        self._init_scaler(x_min=x_min, x_max=x_max, clip_lo=clip_lo, clip_hi=clip_hi)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        x01 = self.scale(x_raw)
        return self.hinge(x01)

    @torch.no_grad()
    def export_params(self):
        exp = self.hinge.export_params()
        return {
            "hinge": {
                "kind": exp.kind,
                "b": exp.b,
                "a": exp.a,
                "t": exp.t.tolist(),
                "c": exp.c.tolist(),
            },
            "scaler": self._export_scaler_params(),
        }