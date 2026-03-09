import torch
import torch.nn as nn
import numpy as np


class CLOSpreadModel(nn.Module):
    """
    Structural additive hinge model for CLO spread (DM) prediction.

    DM = f_base_mvoc(MVOC)
       + Delta_bucket(MVOC, bucket_idx)
       + f_idx(AsOfLevLoanIndex)
       + f_wap(WAP)
       + f_cpn(CpnSpread)
       + bias

    Centering projection keeps additive components zero-mean.
    Combined monotonicity (base + adjustment) enforced via penalty.
    """

    def __init__(
        self,
        mvoc_base: nn.Module,
        mvoc_adj_hinges: list[nn.Module],
        idx_hinge: nn.Module,
        wap_hinge: nn.Module,
        cpn_hinge: nn.Module,
    ):
        super().__init__()
        self.mvoc_base = mvoc_base
        self.mvoc_adj = nn.ModuleList(mvoc_adj_hinges)
        self.idx_hinge = idx_hinge
        self.wap_hinge = wap_hinge
        self.cpn_hinge = cpn_hinge
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        mvoc: torch.Tensor,
        bucket_idx: torch.Tensor,
        lev_idx: torch.Tensor,
        wap: torch.Tensor,
        cpnspread: torch.Tensor,
    ) -> torch.Tensor:
        bucket_idx = bucket_idx.view(-1)
        out = self.mvoc_base(mvoc)

        for b in torch.unique(bucket_idx):
            mask = (bucket_idx == b)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            out[idx] += self.mvoc_adj[b](mvoc[mask])

        out += self.idx_hinge(lev_idx) + self.wap_hinge(wap) + self.cpn_hinge(cpnspread) + self.bias
        return out

    def monotonicity_penalty(self, direction: str = "decreasing", n_eval: int = 50) -> torch.Tensor:
        """Penalise monotonicity violations in (base + adj) for each bucket."""
        grid = torch.linspace(
            float(self.mvoc_base.x_min),
            float(self.mvoc_base.x_max),
            n_eval,
            device=self.bias.device,
        )
        base_vals = self.mvoc_base(grid).squeeze()

        penalty = torch.tensor(0.0, device=self.bias.device)
        for adj in self.mvoc_adj:
            combined = base_vals + adj(grid).squeeze()
            diffs = combined[1:] - combined[:-1]
            if direction == "decreasing":
                penalty = penalty + torch.relu(diffs).mean()
            else:
                penalty = penalty + torch.relu(-diffs).mean()

        return penalty / len(self.mvoc_adj)

    def centre_projection(
        self,
        x_mvoc: torch.Tensor,
        x_idx: torch.Tensor,
        x_wap: torch.Tensor,
        x_cpn: torch.Tensor,
    ) -> None:
        """Projected centering: shift each component to zero data-mean, absorb into bias."""
        for comp, x in [
            (self.mvoc_base, x_mvoc),
            (self.idx_hinge, x_idx),
            (self.wap_hinge, x_wap),
            (self.cpn_hinge, x_cpn),
        ]:
            mean_val = comp(x).mean()
            comp.b -= mean_val
            self.bias += mean_val


class BasisDelta(nn.Module):
    """
    Per-day, per-bucket MVOC adjustment using fixed basis functions.

    delta_{t,b}(m) = w_shift                         (parallel shift)
                   + w_tilt  * (z_bar - z)            (asymmetric tilt)
                   + sum_k w_bend_k * relu(z_k - z)   (bends at user-specified knots)

    Basis functions are fixed (non-learnable); only the per-day per-bucket
    weights are trained.  Knots should be placed densely in the sensitive
    MVOC region (e.g. [0.99, 1.00, 1.01, 1.02]).

    Subsumes ShiftTiltDelta as the special case with bend_knots=[].
    """

    def __init__(
        self,
        n_days: int,
        n_buckets: int,
        mvoc_lo: float,
        mvoc_hi: float,
        mvoc_mean: float,
        bend_knots: np.ndarray | list[float] | None = None,
    ):
        super().__init__()
        self.n_days = n_days
        self.n_buckets = n_buckets

        span = max(mvoc_hi - mvoc_lo, 1e-12)
        self.register_buffer('mvoc_lo', torch.tensor(mvoc_lo, dtype=torch.float32))
        self.register_buffer('mvoc_hi', torch.tensor(mvoc_hi, dtype=torch.float32))
        self.register_buffer('z_bar', torch.tensor((mvoc_mean - mvoc_lo) / span, dtype=torch.float32))

        n_bends = 0
        if bend_knots is not None and len(bend_knots) > 0:
            knots_01 = (np.asarray(bend_knots, dtype=np.float64) - mvoc_lo) / span
            self.register_buffer('bend_z', torch.tensor(knots_01, dtype=torch.float32))
            n_bends = len(knots_01)
        else:
            self.register_buffer('bend_z', torch.zeros(0))

        self.n_bases = 2 + n_bends
        self.weights = nn.Parameter(torch.zeros(n_days, n_buckets, self.n_bases))

    def _build_bases(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate all basis functions at normalised MVOC z.  Returns (N, n_bases)."""
        bases = [
            torch.ones_like(z),
            self.z_bar - z,
        ]
        for k in range(self.bend_z.shape[0]):
            bases.append(torch.relu(self.bend_z[k] - z))
        return torch.cat(bases, dim=1)

    def forward(
        self,
        mvoc: torch.Tensor,
        day_idx: torch.Tensor,
        bucket_idx: torch.Tensor,
    ) -> torch.Tensor:
        d = day_idx.view(-1)
        b = bucket_idx.view(-1)
        mvoc_col = mvoc.view(-1, 1) if mvoc.ndim == 1 else mvoc
        span = (self.mvoc_hi - self.mvoc_lo).clamp(min=1e-12)
        z = ((mvoc_col - self.mvoc_lo) / span).clamp(0.0, 1.0)

        bases = self._build_bases(z)
        w = self.weights[d, b]
        return (bases * w).sum(dim=1, keepdim=True)

    def time_penalty(self, lambda_smooth: float = 1.0, lambda_shrink: float = 0.1) -> torch.Tensor:
        """Temporal smoothness + shrinkage to zero on all basis weights."""
        smooth = (self.weights[1:] - self.weights[:-1]).pow(2).mean()
        shrink = self.weights.pow(2).mean()
        return lambda_smooth * smooth + lambda_shrink * shrink

    @torch.no_grad()
    def compute_effect(
        self,
        mvoc: torch.Tensor,
        day_idx: torch.Tensor,
        bucket_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample delta values (for visualization)."""
        return self.forward(mvoc, day_idx, bucket_idx)
