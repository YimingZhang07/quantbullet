"""Diagnostic utilities for the layered CLO spread model.

All methods read from model memory only -- no raw data needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .layered_spread_model import LayeredSpreadModel


class SpreadModelDiagnostics:
    """Inspect internals of a fitted ``LayeredSpreadModel``.

    Parameters
    ----------
    model : LayeredSpreadModel (fitted)
    """

    def __init__(self, model: LayeredSpreadModel):
        self.model = model
        self._bm = model.bond_memory
        self._kd = model.kernel_delta
        self._idx_to_date = {v: k for k, v in model._date_to_idx.items()} if model._date_to_idx else {}
        self._date_to_idx = model._date_to_idx or {}

    # ------------------------------------------------------------------
    # Bond Memory
    # ------------------------------------------------------------------

    def inspect_bond(self, bond_name: str) -> dict:
        """Current state for a single bond.

        Returns dict with keys: ema, eff_n, shrinkage, last_day_idx,
        last_date, n_observations.
        """
        if bond_name not in self._bm._bond_state:
            return {"error": f"Bond '{bond_name}' not found in memory"}

        ema, eff_n, last_day = self._bm._bond_state[bond_name]
        shrinkage = eff_n / (eff_n + self._bm.k) if eff_n > 0 else 0.0
        last_date = self._idx_to_date.get(last_day)
        n_obs = len(self._bm._bond_history.get(bond_name, []))

        return {
            "bond": bond_name,
            "ema": ema,
            "eff_n": eff_n,
            "shrinkage": shrinkage,
            "adjustment": shrinkage * ema,
            "last_day_idx": last_day,
            "last_date": last_date,
            "n_observations": n_obs,
        }

    def bond_history(self, bond_name: str) -> pd.DataFrame:
        """Full EMA trajectory for a bond from stored history.

        Columns: day_idx, date, residual, ema_before, eff_n, shrinkage, adjustment.
        """
        history = self._bm._bond_history.get(bond_name, [])
        if not history:
            print(f"No history for '{bond_name}'")
            return pd.DataFrame()

        df = pd.DataFrame(history)
        if self._idx_to_date:
            df["date"] = df["day_idx"].map(self._idx_to_date)
        return df

    def bond_summary(self, top_n: int = 20) -> pd.DataFrame:
        """Summary of all bonds: current EMA, shrinkage, n_obs, sorted by abs(adjustment)."""
        rows = []
        for bond, (ema, eff_n, last_day) in self._bm._bond_state.items():
            shrinkage = eff_n / (eff_n + self._bm.k) if eff_n > 0 else 0.0
            rows.append({
                "bond": bond,
                "ema": ema,
                "eff_n": eff_n,
                "shrinkage": shrinkage,
                "adjustment": shrinkage * ema,
                "n_obs": len(self._bm._bond_history.get(bond, [])),
                "last_day_idx": last_day,
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["abs_adj"] = df["adjustment"].abs()
        return df.sort_values("abs_adj", ascending=False).head(top_n).drop(columns="abs_adj")

    def plot_bond_trajectory(self, bond_name: str, figsize=(14, 5)) -> plt.Figure:
        """Plot EMA convergence and adjustment over time for a bond."""
        hist = self.bond_history(bond_name)
        if hist.empty:
            return None

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        x = hist["date"] if "date" in hist.columns else hist["day_idx"]

        axes[0].bar(range(len(hist)), hist["residual"], alpha=0.5, color="tab:gray", label="residual")
        axes[0].plot(range(len(hist)), hist["ema_before"], "tab:blue", linewidth=2, label="EMA (before)")
        axes[0].plot(range(len(hist)), hist["adjustment"], "tab:red", linewidth=2, label="adjustment")
        axes[0].axhline(0, color="black", linewidth=0.5)
        axes[0].set(title=f"{bond_name} — EMA trajectory", ylabel="bps")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(range(len(hist)), hist["shrinkage"], "tab:green", linewidth=2)
        axes[1].set(title="Shrinkage weight", ylabel="weight", ylim=(-0.05, 1.05))
        axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(range(len(hist)), hist["eff_n"], "tab:purple", linewidth=2)
        axes[2].set(title="Effective n_obs", ylabel="count")
        axes[2].grid(True, alpha=0.3)

        for ax in axes:
            ax.set_xlabel("observation #")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Kernel Delta
    # ------------------------------------------------------------------

    def inspect_date(self, day_idx: int | None = None, date: str | None = None) -> pd.DataFrame:
        """Summary of delta curves for a given day.

        Returns one row per bucket: day_idx, bucket, curve_mean, curve_min, curve_max.
        """
        if date is not None and self._date_to_idx:
            ts = pd.Timestamp(date)
            if ts not in self._date_to_idx:
                print(f"Date '{date}' not found")
                return pd.DataFrame()
            day_idx = self._date_to_idx[ts]

        if day_idx is None:
            day_idx = max(self._idx_to_date) if self._idx_to_date else 0

        rows = []
        for b in range(self._kd._n_buckets):
            curve = self._kd._delta_curves.get((day_idx, b))
            if curve is not None:
                rows.append({
                    "day_idx": day_idx,
                    "bucket": b,
                    "curve_mean": float(curve.mean()),
                    "curve_min": float(curve.min()),
                    "curve_max": float(curve.max()),
                    "curve_range": float(curve.max() - curve.min()),
                })
        return pd.DataFrame(rows)

    def plot_delta_curves(
        self, day_idx: int | None = None, date: str | None = None,
        bucket_names: list[str] | None = None,
        figsize=(12, 5),
    ) -> plt.Figure:
        """Plot delta curves for each bucket on a given day."""
        if date is not None and self._date_to_idx:
            ts = pd.Timestamp(date)
            if ts not in self._date_to_idx:
                print(f"Date '{date}' not found")
                return None
            day_idx = self._date_to_idx[ts]

        if day_idx is None:
            day_idx = max(self._idx_to_date) if self._idx_to_date else 0

        colors = ["tab:red", "tab:orange", "tab:blue", "tab:green",
                   "tab:purple", "tab:brown", "tab:pink", "tab:cyan"]
        grid = self._kd._grid

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        date_label = str(self._idx_to_date.get(day_idx, f"day {day_idx}"))[:10]

        for b in range(self._kd._n_buckets):
            label = bucket_names[b] if bucket_names and b < len(bucket_names) else f"bucket {b}"
            color = colors[b % len(colors)]

            delta = self._kd._delta_curves.get((day_idx, b))
            if delta is None:
                continue
            axes[0].plot(grid, delta, color, linewidth=2, label=label)

            if self._kd._struct_grids is not None:
                combined = self._kd._struct_grids[b] + delta
                axes[1].plot(grid, combined, color, linewidth=2, label=label)

        axes[0].axhline(0, color="gray", linewidth=0.5)
        axes[0].set(xlabel="MVOC", ylabel="bps", title=f"Delta curves — {date_label}")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set(xlabel="MVOC", ylabel="bps", title=f"Combined (structural + delta) — {date_label}")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
