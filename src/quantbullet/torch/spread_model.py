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
       + f_nav(EquityNAV)
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
        nav_hinge: nn.Module,
    ):
        super().__init__()
        self.mvoc_base = mvoc_base
        self.mvoc_adj = nn.ModuleList(mvoc_adj_hinges)
        self.idx_hinge = idx_hinge
        self.wap_hinge = wap_hinge
        self.cpn_hinge = cpn_hinge
        self.nav_hinge = nav_hinge
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        mvoc: torch.Tensor,
        bucket_idx: torch.Tensor,
        lev_idx: torch.Tensor,
        wap: torch.Tensor,
        cpnspread: torch.Tensor,
        equity_nav: torch.Tensor,
    ) -> torch.Tensor:
        bucket_idx = bucket_idx.view(-1)
        out = self.mvoc_base(mvoc)

        for b in torch.unique(bucket_idx):
            mask = (bucket_idx == b)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            out[idx] += self.mvoc_adj[b](mvoc[mask])

        out += (self.idx_hinge(lev_idx) + self.wap_hinge(wap)
                + self.cpn_hinge(cpnspread) + self.nav_hinge(equity_nav) + self.bias)
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
        x_nav: torch.Tensor,
    ) -> None:
        """Projected centering: shift each component to zero data-mean, absorb into bias."""
        for comp, x in [
            (self.mvoc_base, x_mvoc),
            (self.idx_hinge, x_idx),
            (self.wap_hinge, x_wap),
            (self.cpn_hinge, x_cpn),
            (self.nav_hinge, x_nav),
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


class GaussianBasisDelta(nn.Module):
    """
    Per-day, per-bucket MVOC adjustment using local Gaussian RBF bases.

    delta_{t,b}(m) = w_shift + sum_k w_k * exp(-(z - z_k)^2 / (2 * sigma_k^2))

    Each Gaussian bump is centred at a user-specified MVOC knot and has
    local support (~3 sigma).  Far from all knots the delta decays to
    w_shift, avoiding the extrapolation problem of relu-based approaches.

    Parameters
    ----------
    centers : array-like
        MVOC values for the RBF centres (actual feature space, not normalised).
    sigma : float or array-like
        Width of each Gaussian.  Scalar → same width for all.  In MVOC units.
    """

    def __init__(
        self,
        n_days: int,
        n_buckets: int,
        mvoc_lo: float,
        mvoc_hi: float,
        centers: np.ndarray | list[float],
        sigma: float | np.ndarray | list[float] = 0.01,
    ):
        super().__init__()
        self.n_days = n_days
        self.n_buckets = n_buckets

        span = max(mvoc_hi - mvoc_lo, 1e-12)
        self.register_buffer('mvoc_lo', torch.tensor(mvoc_lo, dtype=torch.float32))
        self.register_buffer('mvoc_hi', torch.tensor(mvoc_hi, dtype=torch.float32))

        centers_01 = (np.asarray(centers, dtype=np.float64) - mvoc_lo) / span
        self.register_buffer('centers_z', torch.tensor(centers_01, dtype=torch.float32))

        if isinstance(sigma, (int, float)):
            sigma_01 = np.full(len(centers_01), float(sigma) / span)
        else:
            sigma_01 = np.asarray(sigma, dtype=np.float64) / span
        self.register_buffer('sigma_z', torch.tensor(sigma_01, dtype=torch.float32))

        self.n_rbf = len(centers_01)
        self.n_bases = 1 + self.n_rbf
        self.weights = nn.Parameter(torch.zeros(n_days, n_buckets, self.n_bases))

    def _build_bases(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate bases at normalised MVOC z.  Returns (N, 1 + n_rbf)."""
        bases = [torch.ones_like(z)]
        for k in range(self.n_rbf):
            diff = (z - self.centers_z[k]) / self.sigma_z[k]
            bases.append(torch.exp(-0.5 * diff.pow(2)))
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

    def time_penalty(
        self,
        lambda_smooth: float = 1.0,
        lambda_shift: float = 1.0,
        lambda_bump: float = 0.1,
    ) -> torch.Tensor:
        """Temporal smoothness + split shrinkage (heavier on shift, lighter on bumps)."""
        smooth = (self.weights[1:] - self.weights[:-1]).pow(2).mean()
        shift_shrink = self.weights[:, :, 0].pow(2).mean()
        bump_shrink = self.weights[:, :, 1:].pow(2).mean() if self.n_rbf > 0 else 0.0
        return lambda_smooth * smooth + lambda_shift * shift_shrink + lambda_bump * bump_shrink

    @torch.no_grad()
    def compute_effect(
        self,
        mvoc: torch.Tensor,
        day_idx: torch.Tensor,
        bucket_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample delta values (for visualization)."""
        return self.forward(mvoc, day_idx, bucket_idx)


class KernelSmoothDelta:
    """
    Non-parametric daily MVOC curve adjustment via kernel-smoothed residuals.

    For each day-bucket, computes residuals at observed MVOC points, then
    uses Nadaraya-Watson kernel regression to interpolate a smooth delta
    curve over the full MVOC range.  Optionally enforces monotonicity
    via isotonic regression on the combined (structural + delta) curve.

    No nn.Module, no optimisation, no training -- pure closed-form.

    Parameters
    ----------
    bandwidth : float
        Gaussian kernel width in MVOC units.  Controls smoothness.
    alpha : float
        Temporal blending: delta = alpha * today + (1-alpha) * yesterday.
        1.0 = each day independent.
    enforce_monotone : bool
        If True, isotonic regression ensures combined curve is decreasing.
    """

    def __init__(self, bandwidth: float = 0.01, bandwidth_local: float | None = None,
                 alpha: float = 1.0, enforce_monotone: bool = True):
        self.bandwidth = bandwidth
        self.bandwidth_local = bandwidth_local
        self.alpha = alpha
        self.enforce_monotone = enforce_monotone
        self._delta_curves = {}
        self._grid = None

    def fit(
        self,
        structural_model,
        df,
        target_col: str,
        mvoc_col: str = 'MVOC',
        day_col: str = 'day_idx',
        bucket_col: str = 'bucket',
        feature_cols: dict | None = None,
        n_grid: int = 200,
        device=None,
    ):
        """
        Fit daily delta curves for all day-bucket pairs (closed-form).

        Parameters
        ----------
        structural_model : CLOSpreadModel (frozen)
        df : pd.DataFrame with target, MVOC, day_idx, bucket columns
        target_col : target column name
        feature_cols : dict mapping forward() arg names to df column names,
            e.g. {'lev_idx': 'AsOfLevLoanIndex', 'wap': 'WAP', 'cpnspread': 'CpnSpread'}
        """
        if device is None:
            device = next(structural_model.parameters()).device

        structural_model.eval()
        mvoc_lo = float(structural_model.mvoc_base.x_min)
        mvoc_hi = float(structural_model.mvoc_base.x_max)
        self._grid = np.linspace(mvoc_lo, mvoc_hi, n_grid)

        with torch.no_grad():
            grid_t = torch.tensor(self._grid, dtype=torch.float32, device=device)
            base_grid = structural_model.mvoc_base(grid_t).cpu().numpy().ravel()
            n_buckets = len(structural_model.mvoc_adj)
            struct_grids = []
            for b in range(n_buckets):
                adj_grid = structural_model.mvoc_adj[b](grid_t).cpu().numpy().ravel()
                struct_grids.append(base_grid + adj_grid)
        self._struct_grids = struct_grids
        self._n_buckets = n_buckets

        if feature_cols is None:
            feature_cols = {}
        with torch.no_grad():
            mvoc_t = torch.tensor(df[mvoc_col].values, dtype=torch.float32, device=device)
            bucket_t = torch.tensor(df[bucket_col].values, dtype=torch.long, device=device)
            feat_kwargs = {}
            for arg_name, col_name in feature_cols.items():
                feat_kwargs[arg_name] = torch.tensor(df[col_name].values, dtype=torch.float32, device=device)
            struct_pred = structural_model(mvoc_t, bucket_t, **feat_kwargs).cpu().numpy().ravel()

        residuals = df[target_col].values - struct_pred
        mvoc_np = df[mvoc_col].values
        day_np = df[day_col].values.astype(int)
        bucket_np = df[bucket_col].values.astype(int)

        n_days = int(day_np.max()) + 1
        self._n_days = n_days

        for t in range(n_days):
            for b in range(n_buckets):
                mask = (day_np == t) & (bucket_np == b)

                if mask.sum() == 0:
                    self._delta_curves[(t, b)] = self._delta_curves.get((t - 1, b), np.zeros(n_grid))
                    continue

                obs_m, obs_r = mvoc_np[mask], residuals[mask]

                bw_saved = self.bandwidth
                raw_delta = self._nadaraya_watson(self._grid, obs_m, obs_r)

                if self.bandwidth_local is not None:
                    pass1_at_obs = np.interp(obs_m, self._grid, raw_delta)
                    resid2 = obs_r - pass1_at_obs
                    self.bandwidth = self.bandwidth_local
                    raw_delta = raw_delta + self._nadaraya_watson(self._grid, obs_m, resid2)
                    self.bandwidth = bw_saved

                if self.alpha < 1.0 and (t - 1, b) in self._delta_curves:
                    raw_delta = self.alpha * raw_delta + (1 - self.alpha) * self._delta_curves[(t - 1, b)]

                if self.enforce_monotone:
                    combined = struct_grids[b] + raw_delta
                    combined_mono = self._isotonic_decreasing(self._grid, combined)
                    raw_delta = combined_mono - struct_grids[b]

                self._delta_curves[(t, b)] = raw_delta

    def _nadaraya_watson(self, grid, obs_x, obs_y):
        """Kernel-weighted average with confidence decay to 0 far from data."""
        diff = grid[:, None] - obs_x[None, :]
        weights = np.exp(-0.5 * (diff / self.bandwidth) ** 2)
        w_sum = weights.sum(axis=1)
        confidence = 1.0 - np.exp(-w_sum)
        nw = (weights * obs_y[None, :]).sum(axis=1) / w_sum.clip(min=1e-12)
        return confidence * nw

    def _local_linear(self, grid, obs_x, obs_y):
        """Local linear regression: fits a weighted least-squares line at each grid point."""
        diff = grid[:, None] - obs_x[None, :]
        weights = np.exp(-0.5 * (diff / self.bandwidth) ** 2)

        result = np.zeros(len(grid))
        for i in range(len(grid)):
            w = weights[i]
            if w.sum() < 1e-12:
                result[i] = 0.0
                continue
            d = obs_x - grid[i]
            W = np.diag(w)
            X = np.column_stack([np.ones(len(obs_x)), d])
            XtW = X.T @ W
            try:
                beta = np.linalg.solve(XtW @ X + 1e-10 * np.eye(2), XtW @ obs_y)
            except np.linalg.LinAlgError:
                beta = np.array([w @ obs_y / w.sum(), 0.0])
            result[i] = beta[0]
        return result

    @staticmethod
    def _isotonic_decreasing(x, y):
        from sklearn.isotonic import IsotonicRegression
        return IsotonicRegression(increasing=False, out_of_bounds='clip').fit_transform(x, y)

    def forward(self, mvoc, day_idx, bucket_idx):
        if self._grid is None:
            return torch.zeros(mvoc.shape[0], 1, device=mvoc.device)
        mvoc_np = mvoc.detach().cpu().numpy().ravel()
        day_np = day_idx.detach().cpu().numpy().ravel().astype(int)
        bucket_np = bucket_idx.detach().cpu().numpy().ravel().astype(int)
        result = np.zeros(len(mvoc_np))
        for t in np.unique(day_np):
            for b in np.unique(bucket_np):
                mask = (day_np == t) & (bucket_np == b)
                if not mask.any():
                    continue
                curve = self._delta_curves.get((t, b), np.zeros(len(self._grid)))
                result[mask] = np.interp(mvoc_np[mask], self._grid, curve)
        return torch.tensor(result, dtype=torch.float32, device=mvoc.device).view(-1, 1)

    def compute_effect(self, mvoc, day_idx, bucket_idx):
        return self.forward(mvoc, day_idx, bucket_idx)

    def __call__(self, mvoc, day_idx, bucket_idx):
        return self.forward(mvoc, day_idx, bucket_idx)
