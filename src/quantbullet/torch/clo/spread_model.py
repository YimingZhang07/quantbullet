import torch
import torch.nn as nn
import numpy as np


class CLOSpreadModel(nn.Module):
    """
    Structural additive model for CLO spread (DM) prediction.

    DM = f_base_mvoc(MVOC)
       + Delta_bucket(MVOC, bucket_idx)
       + sum_k component_k(feature_k)
       + bias

    Components can be any ``nn.Module`` that maps a 1-D input tensor to a
    (N, 1) output -- including ``Hinge`` (continuous) and ``nn.Embedding``
    (categorical).  They are stored in an ``nn.ModuleDict`` keyed by name.

    Parameters
    ----------
    mvoc_base : nn.Module
        Shared monotone-decreasing MVOC base curve.
    mvoc_adj_hinges : list[nn.Module]
        Per-bucket MVOC shape adjustments.
    components : dict[str, nn.Module]
        Arbitrary additive components keyed by name.
    """

    def __init__(
        self,
        mvoc_base: nn.Module,
        mvoc_adj_hinges: list[nn.Module],
        components: dict[str, nn.Module] | None = None,
    ):
        super().__init__()
        self.mvoc_base = mvoc_base
        self.mvoc_adj = nn.ModuleList(mvoc_adj_hinges)
        self.components = nn.ModuleDict(components or {})
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        mvoc: torch.Tensor,
        bucket_idx: torch.Tensor,
        features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        bucket_idx = bucket_idx.view(-1)
        out = self.mvoc_base(mvoc)

        for b in torch.unique(bucket_idx):
            mask = (bucket_idx == b)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            out[idx] += self.mvoc_adj[b](mvoc[mask])

        for key, comp in self.components.items():
            out = out + comp(features[key])

        out = out + self.bias
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
        features: dict[str, torch.Tensor],
    ) -> None:
        """Projected centering: shift each component to zero data-mean, absorb into bias."""
        self._centre_one(self.mvoc_base, x_mvoc)
        for key, comp in self.components.items():
            self._centre_one(comp, features[key])

    def _centre_one(self, comp: nn.Module, x: torch.Tensor) -> None:
        mean_val = comp(x).mean()
        if hasattr(comp, "b"):
            comp.b -= mean_val
        elif hasattr(comp, "weight"):
            comp.weight.data -= mean_val
        self.bias.data += mean_val


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
        df,
        target_col: str,
        mvoc_col: str = 'MVOC',
        day_col: str = 'day_idx',
        bucket_col: str = 'bucket',
        n_grid: int = 200,
        pred_col: str | None = None,
        struct_grids: list[np.ndarray] | None = None,
        mvoc_range: tuple[float, float] | None = None,
        structural_model: CLOSpreadModel | None = None,
        feature_cols: dict[str, str] | None = None,
        device=None,
    ):
        """
        Fit daily delta curves for all day-bucket pairs (closed-form).

        Parameters
        ----------
        df : pd.DataFrame with target, MVOC, day_idx, bucket columns
        target_col : target column name
        pred_col : column with pre-computed structural predictions.
            When provided, ``structural_model`` / ``feature_cols`` are ignored
            for computing residuals.
        struct_grids : list of per-bucket MVOC curves on a grid (numpy arrays),
            used for monotonicity enforcement.  When None, built from
            ``structural_model`` (which must then be provided).
        mvoc_range : ``(lo, hi)`` for the MVOC evaluation grid.  Required when
            ``struct_grids`` is provided without ``structural_model``.
        structural_model : CLOSpreadModel (used only if *pred_col* / *struct_grids*
            are not provided).
        feature_cols : dict mapping component keys to df column names
            (used only if *pred_col* is None).
        """
        if pred_col is not None:
            struct_pred = df[pred_col].values
        elif structural_model is not None:
            if device is None:
                device = next(structural_model.parameters()).device
            structural_model.eval()
            with torch.no_grad():
                mvoc_t = torch.tensor(df[mvoc_col].values, dtype=torch.float32, device=device)
                bucket_t = torch.tensor(df[bucket_col].values, dtype=torch.long, device=device)
                feat_dict = {}
                for key, col_name in (feature_cols or {}).items():
                    comp = structural_model.components[key]
                    dtype = torch.long if isinstance(comp, nn.Embedding) else torch.float32
                    feat_dict[key] = torch.tensor(df[col_name].values, dtype=dtype, device=device)
                struct_pred = structural_model(mvoc_t, bucket_t, feat_dict).cpu().numpy().ravel()
        else:
            raise ValueError("Provide either pred_col or structural_model")

        if struct_grids is not None:
            if mvoc_range is None:
                raise ValueError("mvoc_range is required when struct_grids is provided")
            mvoc_lo, mvoc_hi = mvoc_range
            self._grid = np.linspace(mvoc_lo, mvoc_hi, n_grid)
            self._struct_grids = struct_grids
            self._n_buckets = len(struct_grids)
        elif structural_model is not None:
            if device is None:
                device = next(structural_model.parameters()).device
            mvoc_lo = float(structural_model.mvoc_base.x_min)
            mvoc_hi = float(structural_model.mvoc_base.x_max)
            self._grid = np.linspace(mvoc_lo, mvoc_hi, n_grid)
            structural_model.eval()
            with torch.no_grad():
                grid_t = torch.tensor(self._grid, dtype=torch.float32, device=device)
                base_grid = structural_model.mvoc_base(grid_t).cpu().numpy().ravel()
                n_buckets = len(structural_model.mvoc_adj)
                sg = []
                for b in range(n_buckets):
                    adj_grid = structural_model.mvoc_adj[b](grid_t).cpu().numpy().ravel()
                    sg.append(base_grid + adj_grid)
            self._struct_grids = sg
            self._n_buckets = n_buckets
        else:
            if mvoc_range is None:
                mvoc_lo, mvoc_hi = float(df[mvoc_col].min()), float(df[mvoc_col].max())
            else:
                mvoc_lo, mvoc_hi = mvoc_range
            self._grid = np.linspace(mvoc_lo, mvoc_hi, n_grid)
            self._struct_grids = None
            self._n_buckets = int(df[bucket_col].max()) + 1

        residuals = df[target_col].values - struct_pred
        mvoc_np = df[mvoc_col].values
        day_np = df[day_col].values.astype(int)
        bucket_np = df[bucket_col].values.astype(int)

        n_days = int(day_np.max()) + 1
        self._n_days = n_days

        for t in range(n_days):
            for b in range(self._n_buckets):
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

                if self.enforce_monotone and self._struct_grids is not None:
                    combined = self._struct_grids[b] + raw_delta
                    combined_mono = self._isotonic_decreasing(self._grid, combined)
                    raw_delta = combined_mono - self._struct_grids[b]

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

    def update(
        self,
        df,
        target_col: str,
        pred_col: str,
        mvoc_col: str = "MVOC",
        day_col: str = "day_idx",
        bucket_col: str = "bucket",
    ) -> None:
        """Incrementally fit delta curves for new day-bucket pairs.

        Appends new curves to ``_delta_curves`` without touching existing ones.
        Requires that ``fit()`` has been called first (to establish the grid
        and struct_grids).
        """
        if self._grid is None:
            raise RuntimeError("Call fit() before update()")

        residuals = df[target_col].values - df[pred_col].values
        mvoc_np = df[mvoc_col].values
        day_np = df[day_col].values.astype(int)
        bucket_np = df[bucket_col].values.astype(int)

        new_days = sorted(set(day_np))
        n_grid = len(self._grid)

        for t in new_days:
            for b in range(self._n_buckets):
                mask = (day_np == t) & (bucket_np == b)

                if mask.sum() == 0:
                    prev = self._delta_curves.get((t - 1, b), np.zeros(n_grid))
                    self._delta_curves[(t, b)] = prev
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

                if self.enforce_monotone and self._struct_grids is not None:
                    combined = self._struct_grids[b] + raw_delta
                    combined_mono = self._isotonic_decreasing(self._grid, combined)
                    raw_delta = combined_mono - self._struct_grids[b]

                self._delta_curves[(t, b)] = raw_delta

        self._n_days = max(self._n_days, max(new_days) + 1)
        print(f"  KernelSmoothDelta: updated {len(new_days)} new days")

    def compute_effect(self, mvoc, day_idx, bucket_idx):
        return self.forward(mvoc, day_idx, bucket_idx)

    def __call__(self, mvoc, day_idx, bucket_idx):
        return self.forward(mvoc, day_idx, bucket_idx)


class BondMemory:
    """
    Per-bond idiosyncratic adjustment via EMA of structural residuals with
    shrinkage toward zero and time decay.

    For each bond, maintains an exponential moving average of the residual
    (actual - structural_pred).  The reported adjustment is shrunk by
    ``effective_n / (effective_n + k)`` so bonds with few (or stale)
    observations stay close to the population model.

    When a bond reappears after a gap, both the EMA and the effective
    observation count are decayed by ``exp(-gap_days / halflife)``.

    **Look-ahead free**: the adjustment recorded for each observation uses
    the EMA state *before* that observation updates it.

    Parameters
    ----------
    alpha : float
        EMA decay factor (0, 1].  Higher = more responsive to recent data.
    k : float
        Shrinkage strength.  Higher = need more observations before trusting
        the per-bond estimate.
    halflife : float or None
        Time-decay half-life in days.  After *halflife* days without an
        observation, a bond's EMA and effective count are halved.
        ``None`` disables time decay (legacy behaviour).
    """

    def __init__(self, alpha: float = 0.3, k: float = 5.0, halflife: float | None = 30.0):
        self.alpha = alpha
        self.k = k
        self.halflife = halflife
        self._decay_rate = np.log(2) / halflife if halflife else 0.0
        self._bond_state: dict[str, tuple[float, float, int]] = {}
        self._bond_history: dict[str, list[dict]] = {}
        self._fitted_adj: np.ndarray | None = None
        self._fitted_index = None

    def fit(
        self,
        df,
        target_col: str,
        bond_col: str = "SecurityName",
        day_col: str = "day_idx",
        pred_col: str | None = None,
        structural_model: CLOSpreadModel | None = None,
        feature_cols: dict[str, str] | None = None,
        mvoc_col: str = "MVOC",
        device=None,
    ):
        """Fit per-bond EMA adjustments chronologically.

        Parameters
        ----------
        df : pd.DataFrame
        target_col : target column name
        bond_col : column identifying each bond.
        pred_col : column with pre-computed structural predictions.
            When provided, ``structural_model`` / ``feature_cols`` are ignored.
        structural_model : CLOSpreadModel (used only if *pred_col* is None).
        feature_cols : dict mapping component keys to df column names
            (used only if *pred_col* is None).
        """
        if pred_col is not None:
            struct_pred = df[pred_col].values
        elif structural_model is not None:
            if device is None:
                device = next(structural_model.parameters()).device
            structural_model.eval()
            with torch.no_grad():
                mvoc_t = torch.tensor(df[mvoc_col].values, dtype=torch.float32, device=device)
                bucket_t = torch.tensor(df["bucket"].values, dtype=torch.long, device=device)
                feat_dict = {}
                for key, col_name in (feature_cols or {}).items():
                    comp = structural_model.components[key]
                    dtype = torch.long if isinstance(comp, nn.Embedding) else torch.float32
                    feat_dict[key] = torch.tensor(df[col_name].values, dtype=dtype, device=device)
                struct_pred = structural_model(mvoc_t, bucket_t, feat_dict).cpu().numpy().ravel()
        else:
            raise ValueError("Provide either pred_col or structural_model")

        residuals = df[target_col].values - struct_pred

        order = np.lexsort((df[bond_col].values, df[day_col].values))

        ema: dict[str, float] = {}
        eff_n: dict[str, float] = {}
        last_day: dict[str, int] = {}
        history: dict[str, list[dict]] = {}
        adjustments = np.zeros(len(df))

        bond_vals = df[bond_col].values
        day_vals = df[day_col].values
        for i in order:
            bond = bond_vals[i]
            day = int(day_vals[i])

            current_ema = ema.get(bond, 0.0)
            current_n = eff_n.get(bond, 0.0)

            if bond in last_day and self._decay_rate > 0:
                gap = day - last_day[bond]
                if gap > 0:
                    decay = np.exp(-self._decay_rate * gap)
                    current_ema *= decay
                    current_n *= decay

            shrinkage = current_n / (current_n + self.k) if current_n > 0 else 0.0
            adjustments[i] = shrinkage * current_ema

            history.setdefault(bond, []).append({
                "day_idx": day,
                "residual": float(residuals[i]),
                "ema_before": float(current_ema),
                "eff_n": float(current_n),
                "shrinkage": float(shrinkage),
                "adjustment": float(adjustments[i]),
            })

            new_ema = self.alpha * residuals[i] + (1.0 - self.alpha) * current_ema
            ema[bond] = new_ema
            eff_n[bond] = current_n + 1
            last_day[bond] = day

        self._bond_state = {b: (ema[b], eff_n[b], last_day[b]) for b in ema}
        self._bond_history = history
        self._fitted_adj = adjustments
        self._fitted_index = df.index

        self._max_day = int(day_vals.max())
        n_bonds = len(self._bond_state)
        active = sum(1 for _, (_, n, _) in self._bond_state.items() if n >= self.k)
        print(f"  BondMemory: {n_bonds} bonds, {active} with >= {self.k:.0f} obs (fully active)")

    def fitted_values(self) -> np.ndarray:
        """Per-observation adjustments (look-ahead free), aligned with the df passed to fit()."""
        if self._fitted_adj is None:
            raise RuntimeError("Call fit() first")
        return self._fitted_adj

    def update(
        self,
        df,
        target_col: str,
        pred_col: str,
        bond_col: str = "SecurityName",
        day_col: str = "day_idx",
    ) -> np.ndarray:
        """Incremental update: continue EMA from existing state with new observations.

        Returns per-observation adjustments (look-ahead free) for the new data.
        Updates ``_bond_state`` in place.
        """
        residuals = df[target_col].values - df[pred_col].values
        order = np.lexsort((df[bond_col].values, df[day_col].values))
        adjustments = np.zeros(len(df))

        bond_vals = df[bond_col].values
        day_vals = df[day_col].values

        for i in order:
            bond = bond_vals[i]
            day = int(day_vals[i])

            if bond in self._bond_state:
                current_ema, current_n, last = self._bond_state[bond]
                if self._decay_rate > 0:
                    gap = day - last
                    if gap > 0:
                        decay = np.exp(-self._decay_rate * gap)
                        current_ema *= decay
                        current_n *= decay
            else:
                current_ema, current_n = 0.0, 0.0

            shrinkage = current_n / (current_n + self.k) if current_n > 0 else 0.0
            adjustments[i] = shrinkage * current_ema

            self._bond_history.setdefault(bond, []).append({
                "day_idx": day,
                "residual": float(residuals[i]),
                "ema_before": float(current_ema),
                "eff_n": float(current_n),
                "shrinkage": float(shrinkage),
                "adjustment": float(adjustments[i]),
            })

            new_ema = self.alpha * residuals[i] + (1.0 - self.alpha) * current_ema
            self._bond_state[bond] = (new_ema, current_n + 1, day)

        self._max_day = int(day_vals.max())
        self._fitted_adj = adjustments
        self._fitted_index = df.index
        return adjustments

    def get_adjustment(self, bond_ids, day_idxs) -> np.ndarray:
        """Get the best available adjustment for each (bond, day) pair.

        For (bond, day) pairs seen during fit/update, returns the historical
        look-ahead-free adjustment.  For unseen pairs, falls back to the
        current EMA state with time decay.
        """
        history_lookup: dict[tuple, float] = {}
        for bond, entries in self._bond_history.items():
            for entry in entries:
                history_lookup[(bond, entry["day_idx"])] = entry["adjustment"]

        result = np.zeros(len(bond_ids))
        for i in range(len(bond_ids)):
            bond = bond_ids[i]
            day = int(day_idxs[i])
            key = (bond, day)
            if key in history_lookup:
                result[i] = history_lookup[key]
            elif bond in self._bond_state:
                ema_val, n, last = self._bond_state[bond]
                if self._decay_rate > 0:
                    gap = day - last
                    if gap > 0:
                        decay = np.exp(-self._decay_rate * gap)
                        ema_val *= decay
                        n *= decay
                shrinkage = n / (n + self.k) if n > 0 else 0.0
                result[i] = shrinkage * ema_val
        return result
