"""Production pipeline for the layered CLO spread model.

``LayeredSpreadModel`` wraps the full workflow:

    prediction = structural + bond_memory + kernel_delta

into a single object with ``fit`` / ``predict`` / ``save`` / ``load``.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..hinge import Hinge
from ..utils import freeze, unfreeze, train_model_lbfgs
from ..customized import EpsilonInsensitiveLoss
from ..preprocessing import DataPreprocessor
from .spread_model import CLOSpreadModel, BondMemory, KernelSmoothDelta


@dataclass
class ComponentSpec:
    """Specification for a single additive component."""

    col: str
    type: str = "hinge"
    n_knots: int = 10
    shape: str = "generic"
    monotone: str | None = None
    intercept: bool = False
    n_categories: int | None = None


@dataclass
class LayeredSpreadConfig:
    """All hyperparameters for the layered spread model."""

    target: str = "RiskSpread"
    date_col: str = "AsOfDate"
    bond_col: str = "SecurityName"
    mvoc_col: str = "c_MVOC"
    mvoc_raw_col: str = "MVOC"

    bucket_col: str = "NumMosToReinv"
    bucket_edges: list[float] = field(default_factory=lambda: [-24, 0, 12, 24, 48])
    bucket_names: list[str] = field(default_factory=lambda: ["<-24", "-24\u20130", "0\u201312", "12\u201324", "24\u201348", ">48"])

    mvoc_n_knots: int = 20
    mvoc_adj_n_knots: int = 10

    components: dict[str, ComponentSpec] = field(default_factory=dict)
    feature_cols: dict[str, str] = field(default_factory=dict)

    stage1_freeze: list[str] = field(default_factory=lambda: ["wap", "cpn", "nav", "listing", "tier"])
    lambda_mono: float = 10.0
    loss_epsilon: float = 5.0
    lbfgs_steps: int = 200
    lbfgs_patience: int = 15
    seed: int = 42

    bm_alpha: float = 0.3
    bm_k: float = 5.0
    bm_halflife: float = 15.0

    kd_bandwidth: float = 0.01
    kd_bandwidth_local: float | None = None
    kd_alpha: float = 1.0
    kd_enforce_monotone: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["components"] = {k: asdict(v) for k, v in self.components.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LayeredSpreadConfig":
        d = d.copy()
        d["components"] = {k: ComponentSpec(**v) for k, v in d["components"].items()}
        return cls(**d)


class LayeredSpreadModel:
    """End-to-end layered spread model: preprocess, structural, bond memory, kernel delta.

    Parameters
    ----------
    config : LayeredSpreadConfig
    preprocessor : DataPreprocessor
    """

    def __init__(self, config: LayeredSpreadConfig, preprocessor: DataPreprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.structural_model: CLOSpreadModel | None = None
        self.bond_memory: BondMemory | None = None
        self.kernel_delta: KernelSmoothDelta | None = None

        self._struct_grids: list[np.ndarray] | None = None
        self._mvoc_range: tuple[float, float] | None = None
        self._n_buckets: int = len(config.bucket_names)
        self._date_to_idx: dict = {}
        self._training_losses: list[float] = []

    def fit(self, raw_data: pd.DataFrame) -> "LayeredSpreadModel":
        """Full training pipeline."""
        cfg = self.config
        print("=== Data preprocessing ===")
        df = self.preprocessor.fit_transform(raw_data)
        df = self._prepare_data(df, strict=True)

        print("\n=== Building structural model ===")
        self._build_structural(df)

        print("\n=== Training structural model ===")
        tb, F = self._prepare_tensors(df)
        self._train_structural(tb, F)

        print("\n=== Computing structural predictions ===")
        df["pred_structural"] = self._structural_predict(tb, F)
        self._compute_struct_grids()

        print("\n=== Stage 3a: Bond Memory ===")
        self.bond_memory = BondMemory(
            alpha=cfg.bm_alpha, k=cfg.bm_k, halflife=cfg.bm_halflife,
        )
        self.bond_memory.fit(
            df, target_col=cfg.target, pred_col="pred_structural",
            bond_col=cfg.bond_col, day_col="day_idx",
        )

        print("\n=== Stage 3b: Kernel Delta ===")
        df["_bond_adj"] = self.bond_memory.fitted_values()
        df["_target_after_memory"] = df[cfg.target] - df["_bond_adj"]

        self.kernel_delta = KernelSmoothDelta(
            bandwidth=cfg.kd_bandwidth,
            bandwidth_local=cfg.kd_bandwidth_local,
            alpha=cfg.kd_alpha,
            enforce_monotone=cfg.kd_enforce_monotone,
        )
        self.kernel_delta.fit(
            df, target_col="_target_after_memory", pred_col="pred_structural",
            mvoc_col=cfg.mvoc_col,
            struct_grids=self._struct_grids,
            mvoc_range=self._mvoc_range,
        )

        print("\n=== Training complete ===")
        return self

    def predict(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Score new data with all layers.

        Returns a DataFrame with the **same number of rows** as the input.
        Non-scorable rows get NaN predictions + ``_scorable=False`` with reason.
        """
        df, scored, mask, tb, F = self._check_and_split(raw_data)
        if scored is None:
            return df

        cfg = self.config
        structural = self._structural_predict(tb, F)
        bond_adj = self.bond_memory.get_adjustment(scored[cfg.bond_col].values, scored["day_idx"].values)
        daily = self._get_delta(tb, scored)

        return self._fill_predictions(df, mask, scored, structural, bond_adj, daily)

    def refit_bond_memory_and_delta(
        self,
        raw_data: pd.DataFrame,
        recent_cutoff: str | None = None,
    ) -> None:
        """Refit bond memory + kernel delta from scratch (structural frozen).

        The structural model weights are NOT retrained -- only the two
        closed-form layers are recalibrated.  This is fast (O(N), no
        gradient descent).

        Parameters
        ----------
        raw_data : full dataset (old + new).  Pass the complete history so
            bond memory can build EMA trajectories from the beginning.
        recent_cutoff : if given, only fit layers on data from this date
            onward.  **Caution**: this erases all bond memory and delta
            curves before the cutoff (bonds lose their history, old dates
            lose their curves).  Prefer passing full data and letting
            ``BondMemory.halflife`` handle relevance decay naturally.
            Use this only for speed on very large datasets.
        """
        cfg = self.config
        _, scored, _, tb, F = self._check_and_split(raw_data)
        if scored is None:
            print("=== Refit aborted (no scorable rows) ===")
            return

        scored["pred_structural"] = self._structural_predict(tb, F)

        if recent_cutoff:
            layer_df = scored[scored[cfg.date_col] >= pd.Timestamp(recent_cutoff)].copy()
        else:
            layer_df = scored

        print("=== Refitting Bond Memory ===")
        self.bond_memory = BondMemory(
            alpha=cfg.bm_alpha, k=cfg.bm_k, halflife=cfg.bm_halflife,
        )
        self.bond_memory.fit(
            layer_df, target_col=cfg.target, pred_col="pred_structural",
            bond_col=cfg.bond_col, day_col="day_idx",
        )

        print("=== Refitting Kernel Delta ===")
        layer_df["_bond_adj"] = self.bond_memory.fitted_values()
        layer_df["_target_after_memory"] = layer_df[cfg.target] - layer_df["_bond_adj"]

        self.kernel_delta = KernelSmoothDelta(
            bandwidth=cfg.kd_bandwidth,
            bandwidth_local=cfg.kd_bandwidth_local,
            alpha=cfg.kd_alpha,
            enforce_monotone=cfg.kd_enforce_monotone,
        )
        self.kernel_delta.fit(
            layer_df, target_col="_target_after_memory", pred_col="pred_structural",
            mvoc_col=cfg.mvoc_col,
            struct_grids=self._struct_grids,
            mvoc_range=self._mvoc_range,
        )
        print("=== Refit complete ===")

    def update(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Fast incremental update with new actuals, then predict.

        Continues bond memory EMA from existing state and fits kernel delta
        curves for new day-buckets only.  The structural model is not touched.

        Parameters
        ----------
        raw_data : new observations with actuals (e.g. one day's data).

        Returns
        -------
        pd.DataFrame with all prediction / residual columns.
        """
        df, scored, mask, tb, F = self._check_and_split(raw_data)
        if scored is None:
            print("=== Update complete (no scorable rows) ===")
            return df

        cfg = self.config
        print("=== Incremental update ===")
        structural = self._structural_predict(tb, F)
        scored["pred_structural"] = structural

        bond_adj = self.bond_memory.update(
            scored, target_col=cfg.target, pred_col="pred_structural",
            bond_col=cfg.bond_col, day_col="day_idx",
        )
        scored["_target_after_memory"] = scored[cfg.target] - bond_adj

        self.kernel_delta.update(
            scored, target_col="_target_after_memory", pred_col="pred_structural",
            mvoc_col=cfg.mvoc_col, day_col="day_idx",
        )
        daily = self.kernel_delta(tb.x_mvoc, tb.x_day, tb.x_bucket).cpu().squeeze().numpy()

        print("=== Update complete ===")
        return self._fill_predictions(df, mask, scored, structural, bond_adj, daily)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        """Save full model state to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        self.preprocessor.save(directory / "preprocessor.pkl")

        torch.save(self.structural_model.state_dict(), directory / "structural_model.pt")

        with open(directory / "bond_memory.pkl", "wb") as f:
            pickle.dump({
                "bond_state": self.bond_memory._bond_state,
                "max_day": getattr(self.bond_memory, "_max_day", 0),
            }, f)

        with open(directory / "kernel_delta.pkl", "wb") as f:
            pickle.dump({
                "delta_curves": self.kernel_delta._delta_curves,
                "grid": self.kernel_delta._grid,
                "struct_grids": self.kernel_delta._struct_grids,
                "n_buckets": self.kernel_delta._n_buckets,
                "n_days": self.kernel_delta._n_days,
            }, f)

        with open(directory / "model_state.pkl", "wb") as f:
            pickle.dump({
                "struct_grids": self._struct_grids,
                "mvoc_range": self._mvoc_range,
                "n_buckets": self._n_buckets,
                "date_to_idx": self._date_to_idx,
                "training_losses": self._training_losses,
            }, f)

        print(f"Saved to {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "LayeredSpreadModel":
        """Load a saved pipeline."""
        directory = Path(directory)

        with open(directory / "config.json") as f:
            config = LayeredSpreadConfig.from_dict(json.load(f))

        preprocessor = DataPreprocessor.load(directory / "preprocessor.pkl")

        obj = cls(config, preprocessor)

        obj._build_structural_from_config()
        obj.structural_model.load_state_dict(
            torch.load(directory / "structural_model.pt", map_location=obj.device, weights_only=True)
        )
        obj.structural_model.to(obj.device)
        obj.structural_model.eval()

        with open(directory / "bond_memory.pkl", "rb") as f:
            bm_data = pickle.load(f)
        obj.bond_memory = BondMemory(
            alpha=config.bm_alpha, k=config.bm_k, halflife=config.bm_halflife,
        )
        obj.bond_memory._bond_state = bm_data["bond_state"]
        obj.bond_memory._max_day = bm_data["max_day"]
        obj.bond_memory._fitted_adj = None

        with open(directory / "kernel_delta.pkl", "rb") as f:
            kd_data = pickle.load(f)
        obj.kernel_delta = KernelSmoothDelta(
            bandwidth=config.kd_bandwidth,
            bandwidth_local=config.kd_bandwidth_local,
            alpha=config.kd_alpha,
            enforce_monotone=config.kd_enforce_monotone,
        )
        obj.kernel_delta._delta_curves = kd_data["delta_curves"]
        obj.kernel_delta._grid = kd_data["grid"]
        obj.kernel_delta._struct_grids = kd_data["struct_grids"]
        obj.kernel_delta._n_buckets = kd_data["n_buckets"]
        obj.kernel_delta._n_days = kd_data["n_days"]

        with open(directory / "model_state.pkl", "rb") as f:
            ms = pickle.load(f)
        obj._struct_grids = ms["struct_grids"]
        obj._mvoc_range = ms["mvoc_range"]
        obj._n_buckets = ms["n_buckets"]
        obj._date_to_idx = ms.get("date_to_idx", {})
        obj._training_losses = ms["training_losses"]

        print(f"Loaded from {directory}")
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_data(self, df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
        """Assign bucket and day_idx columns.

        If ``strict=True`` (used by ``fit``), raises on NaN in required columns.
        """
        cfg = self.config
        if strict:
            required = [cfg.target, cfg.mvoc_col, cfg.bucket_col, cfg.date_col]
            required += list(cfg.feature_cols.values())
            present = [c for c in required if c in df.columns]
            na_counts = df[present].isna().sum()
            bad = na_counts[na_counts > 0]
            if len(bad) > 0:
                raise ValueError(
                    f"NaN values found in required columns (clean data before passing to model):\n{bad.to_string()}"
                )

        bins = [-np.inf] + cfg.bucket_edges + [np.inf]
        labels = list(range(len(cfg.bucket_names)))
        df["bucket"] = pd.cut(df[cfg.bucket_col], bins=bins, labels=labels).astype(int)

        new_dates = sorted(set(df[cfg.date_col].unique()) - set(self._date_to_idx))
        next_idx = max(self._date_to_idx.values()) + 1 if self._date_to_idx else 0
        for d in new_dates:
            self._date_to_idx[d] = next_idx
            next_idx += 1
        df["day_idx"] = df[cfg.date_col].map(self._date_to_idx)
        return df

    def _check_and_split(self, raw_data: pd.DataFrame):
        """Transform, check scorability, split into scorable subset.

        Returns (full_df, scored_df, scorable_mask, tb, F).
        ``scored_df`` has buckets + day_idx assigned and tensors ready.
        Returns ``None`` for scored_df/tb/F if no scorable rows.
        """
        df = self.preprocessor.transform(raw_data)
        df = self.preprocessor.check(df)

        scorable_mask = df["_scorable"].values
        n_total, n_scorable = len(df), int(scorable_mask.sum())
        if n_total - n_scorable > 0:
            print(f"  {n_total - n_scorable}/{n_total} rows not scorable")

        if n_scorable == 0:
            return df, None, scorable_mask, None, None

        scored = df.loc[scorable_mask].copy()
        scored = self._prepare_data(scored)
        tb, F = self._prepare_tensors(scored)
        return df, scored, scorable_mask, tb, F

    PRED_COLS = [
        "adj_structural", "adj_bond", "adj_delta",
        "pred_structural", "pred_bond", "pred_final",
        "resid_structural", "resid_bond", "resid_final",
    ]

    def _fill_predictions(
        self, df: pd.DataFrame, mask, scored: pd.DataFrame,
        structural: np.ndarray, bond_adj: np.ndarray, daily: np.ndarray,
    ) -> pd.DataFrame:
        """Write prediction columns into full_df from scored results."""
        target = self.config.target
        for col in self.PRED_COLS:
            if col not in df.columns:
                df[col] = np.nan

        df.loc[mask, "adj_structural"] = structural
        df.loc[mask, "adj_bond"] = bond_adj
        df.loc[mask, "adj_delta"] = daily
        df.loc[mask, "pred_structural"] = structural
        df.loc[mask, "pred_bond"] = structural + bond_adj
        df.loc[mask, "pred_final"] = structural + bond_adj + daily

        if target in df.columns:
            df.loc[mask, "resid_structural"] = df.loc[mask, target].values - structural
            df.loc[mask, "resid_bond"] = df.loc[mask, target].values - (structural + bond_adj)
            df.loc[mask, "resid_final"] = df.loc[mask, target].values - (structural + bond_adj + daily)
        return df

    def _get_delta(self, tb, scored: pd.DataFrame) -> np.ndarray:
        """Get kernel delta: use per-row day_idx when fitted, fall back to last day."""
        fitted_days = set(t for (t, _b) in self.kernel_delta._delta_curves) \
            if self.kernel_delta._delta_curves else set()
        last_day = max(fitted_days) if fitted_days else 0

        day_np = scored["day_idx"].values.copy()
        for i, d in enumerate(day_np):
            if d not in fitted_days:
                day_np[i] = last_day
        day_tensor = torch.tensor(day_np, dtype=torch.long, device=self.device)
        return self.kernel_delta(tb.x_mvoc, day_tensor, tb.x_bucket).cpu().squeeze().numpy()

    def _build_structural(self, df: pd.DataFrame) -> None:
        cfg = self.config
        mvoc_lo = float(df[cfg.mvoc_col].min())
        mvoc_hi = float(df[cfg.mvoc_col].max())
        self._mvoc_range = (mvoc_lo, mvoc_hi)

        mvoc_base = Hinge(
            n_knots=cfg.mvoc_n_knots, x_range=(mvoc_lo, mvoc_hi),
            shape="generic", monotone="decreasing", intercept=True,
        )
        mvoc_adj = [
            Hinge(n_knots=cfg.mvoc_adj_n_knots, x_range=(mvoc_lo, mvoc_hi), shape="generic", intercept=True)
            for _ in range(self._n_buckets)
        ]

        components = {}
        for key, spec in cfg.components.items():
            if spec.type == "embedding":
                components[key] = nn.Embedding(spec.n_categories, 1)
            else:
                col_lo = float(df[spec.col].min())
                col_hi = float(df[spec.col].max())
                components[key] = Hinge(
                    n_knots=spec.n_knots, x_range=(col_lo, col_hi),
                    shape=spec.shape, monotone=spec.monotone, intercept=spec.intercept,
                )

        self.structural_model = CLOSpreadModel(
            mvoc_base=mvoc_base, mvoc_adj_hinges=mvoc_adj, components=components,
        ).to(self.device)

    def _build_structural_from_config(self) -> None:
        """Build structural model architecture for loading weights.

        Uses placeholder ranges (0, 1) since actual ranges are stored in the
        saved weights (x_min / x_max buffers).
        """
        cfg = self.config
        mvoc_base = Hinge(
            n_knots=cfg.mvoc_n_knots, x_range=(0, 1),
            shape="generic", monotone="decreasing", intercept=True,
        )
        mvoc_adj = [
            Hinge(n_knots=cfg.mvoc_adj_n_knots, x_range=(0, 1), shape="generic", intercept=True)
            for _ in range(self._n_buckets)
        ]
        components = {}
        for key, spec in cfg.components.items():
            if spec.type == "embedding":
                components[key] = nn.Embedding(spec.n_categories, 1)
            else:
                components[key] = Hinge(
                    n_knots=spec.n_knots, x_range=(0, 1),
                    shape=spec.shape, monotone=spec.monotone, intercept=spec.intercept,
                )
        self.structural_model = CLOSpreadModel(
            mvoc_base=mvoc_base, mvoc_adj_hinges=mvoc_adj, components=components,
        )

    def _prepare_tensors(self, df: pd.DataFrame):
        cfg = self.config

        x_mvoc = torch.from_numpy(df[cfg.mvoc_col].values).float().to(self.device)
        x_day = torch.from_numpy(df["day_idx"].values).long().to(self.device)
        x_bucket = torch.from_numpy(df["bucket"].values).long().to(self.device)
        y = torch.from_numpy(df[cfg.target].values).float().to(self.device).view(-1, 1)

        features = {}
        for key, col in cfg.feature_cols.items():
            vals = df[col].values
            if np.issubdtype(vals.dtype, np.integer):
                features[key] = torch.from_numpy(vals).long().to(self.device)
            else:
                features[key] = torch.from_numpy(vals).float().to(self.device)

        class _TB:
            pass
        tb = _TB()
        tb.x_mvoc = x_mvoc
        tb.x_day = x_day
        tb.x_bucket = x_bucket
        tb.y = y
        tb.features = features
        return tb, features

    def _train_structural(self, tb, F) -> None:
        cfg = self.config
        model = self.structural_model
        loss_fn = EpsilonInsensitiveLoss(epsilon=cfg.loss_epsilon)

        torch.manual_seed(cfg.seed)

        def forward_fn(_m):
            return model(tb.x_mvoc, tb.x_bucket, F), tb.y

        def composite_loss(pred, target):
            return loss_fn(pred, target) + cfg.lambda_mono * model.monotonicity_penalty()

        def centre_projection():
            model.centre_projection(tb.x_mvoc, F)

        print("  Stage 1: MVOC + bucket adj + unfrozen components")
        for key in cfg.stage1_freeze:
            freeze(model.components[key])

        losses_s1 = train_model_lbfgs(
            model=model, loss_fn=composite_loss, forward_fn=forward_fn,
            steps=cfg.lbfgs_steps, lr=1.0, early_stopping=True,
            patience=cfg.lbfgs_patience, projection_fn=centre_projection,
        )
        print(f"    bias = {model.bias.item():.1f} bps")

        print("  Stage 2: all components (joint)")
        for key in cfg.stage1_freeze:
            unfreeze(model.components[key])

        losses_s2 = train_model_lbfgs(
            model=model, loss_fn=composite_loss, forward_fn=forward_fn,
            steps=cfg.lbfgs_steps, lr=1.0, early_stopping=True,
            patience=cfg.lbfgs_patience, projection_fn=centre_projection,
        )
        print(f"    bias = {model.bias.item():.1f} bps")

        self._training_losses = losses_s1 + losses_s2

    def _structural_predict(self, tb, F) -> np.ndarray:
        self.structural_model.eval()
        with torch.no_grad():
            return self.structural_model(tb.x_mvoc, tb.x_bucket, F).cpu().squeeze().numpy()

    def _compute_struct_grids(self) -> None:
        mvoc_lo, mvoc_hi = self._mvoc_range
        grid_t = torch.linspace(mvoc_lo, mvoc_hi, 200, device=self.device)
        with torch.no_grad():
            base = self.structural_model.mvoc_base(grid_t).cpu().numpy().ravel()
            self._struct_grids = [
                base + self.structural_model.mvoc_adj[b](grid_t).cpu().numpy().ravel()
                for b in range(self._n_buckets)
            ]
