from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quantbullet.dfutils.label import get_bins_and_labels
from quantbullet.plot.cycles import ECONOMIST_COLORS, use_economist_cycle


@dataclass(frozen=True)
class BinnedXYResult:
    """Returned by binned plotting helper for debugging/tests."""
    binned_df: pd.DataFrame
    bin_labels: list[str] | None


def _as_list(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    return list(x)


def _bin_centers_from_intervals(intervals: pd.Series) -> np.ndarray:
    # For finite intervals use midpoint; if infinite, fall back to finite edge.
    centers = []
    for itv in intervals:
        if hasattr(itv, "mid") and np.isfinite(itv.mid):
            centers.append(float(itv.mid))
        else:
            # interval from pd.cut/pd.qcut: use right edge if finite else left
            r = getattr(itv, "right", np.nan)
            l = getattr(itv, "left", np.nan)
            if np.isfinite(r):
                centers.append(float(r))
            elif np.isfinite(l):
                centers.append(float(l))
            else:
                centers.append(np.nan)
    return np.asarray(centers, dtype=float)


def _bin_positions_from_intervals(intervals: pd.Series, *, position: Literal["mid", "left", "right"] = "mid") -> np.ndarray:
    """Get numeric x positions from pandas Interval bins.

    If the requested position is infinite (e.g. left=-inf), fall back to the finite edge.
    """
    xs = []
    for itv in intervals:
        l = getattr(itv, "left", np.nan)
        r = getattr(itv, "right", np.nan)
        mid = getattr(itv, "mid", np.nan)

        if position == "left":
            cand = l
            fallback = r
        elif position == "right":
            cand = r
            fallback = l
        else:
            cand = mid
            # fall back to right then left
            fallback = r if np.isfinite(r) else l

        if np.isfinite(cand):
            xs.append(float(cand))
        elif np.isfinite(fallback):
            xs.append(float(fallback))
        else:
            xs.append(np.nan)
    return np.asarray(xs, dtype=float)


def _prepare_binned_stats(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    *,
    cutoffs: list[float] | np.ndarray | None = None,
    n_bins: int = 10,
    include_inf: bool = False,
    label_style: Literal["simple", "interval"] = "simple",
    right_closed: bool = True,
    bin_x: Literal["mid", "left", "right"] = "mid",
    err: Literal["std", "sem"] = "std",
    band_k: float = 1.0,
) -> BinnedXYResult:
    x = df[x_col]
    X = df[[x_col] + y_cols].copy()

    bin_labels: list[str] | None = None
    if cutoffs is not None:
        bins, bin_labels = get_bins_and_labels(
            cutoffs=cutoffs,
            include_inf=include_inf,
            decimal_places=2,
            label_style=label_style,
            right_closed=right_closed,
        )
        # Keep Interval bins (no labels) so we can compute numeric centers correctly.
        # We'll use `bin_labels` only for tick labels in the plot.
        X["_bin"] = pd.cut(x, bins=bins, right=right_closed, include_lowest=True)
    else:
        X["_bin"] = pd.qcut(x, q=n_bins, duplicates="drop")

    group = X.groupby("_bin", observed=False)

    out = pd.DataFrame(index=group.size().index)
    out["count"] = group.size().astype(int).values

    # x position per bin
    out["x_center"] = _bin_positions_from_intervals(out.index.to_series(), position=bin_x)

    # Optional human-friendly labels for cutoffs-based bins
    if bin_labels is not None:
        cats = list(getattr(X["_bin"].cat, "categories", []))
        if len(cats) == len(bin_labels):
            out["bin_label"] = [bin_labels[cats.index(itv)] for itv in out.index]  # type: ignore[arg-type]
        else:
            out["bin_label"] = [str(itv) for itv in out.index]

    for y in y_cols:
        vals = group[y]
        mu = vals.mean()
        sd = vals.std(ddof=1)
        if err == "sem":
            se = sd / np.sqrt(np.maximum(out["count"].to_numpy(dtype=float), 1.0))
            band = band_k * se
        else:
            band = band_k * sd
        out[f"{y}__mean"] = mu.values
        out[f"{y}__lo"] = (mu - band).values
        out[f"{y}__hi"] = (mu + band).values

    out = out.reset_index(drop=False).rename(columns={"_bin": "bin"})
    return BinnedXYResult(binned_df=out, bin_labels=bin_labels)


def plot_scatter_multi_y(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_cols: str | Iterable[str],
    mode: Literal["scatter", "binned"] = "scatter",
    ax=None,
    figsize: tuple[float, float] = (7, 4),
    alpha: float = 0.35,
    s: float = 12,
    show_points_in_binned: bool = False,
    # binning
    cutoffs: list[float] | np.ndarray | None = None,
    n_bins: int = 10,
    include_inf: bool = False,
    label_style: Literal["simple", "interval"] = "simple",
    right_closed: bool = True,
    bin_x: Literal["mid", "left", "right"] = "mid",
    x_tick_labels: Literal["numeric", "label", "none"] = "numeric",
    x_tick_decimals: int = 2,
    # band
    err: Literal["std", "sem"] = "std",
    band_k: float = 1.0,
    band_alpha: float = 0.15,
    # binned rendering
    binned_style: Literal["errorbar", "band"] = "errorbar",
    connect_means: bool = False,
    capsize: float = 3,
):
    """
    Plot x against multiple y columns.

    Modes
    -----
    - scatter: raw scatter for each y
    - binned: bin by x (cutoffs via get_bins_and_labels, otherwise qcut) and plot mean with uncertainty.
      Default style is categorical-like: mean dot + vertical whiskers (like GAM factor plots).
    """
    y_cols_l = _as_list(y_cols)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Ensure we use your Economist palette/cycle without hardcoding colors here.
    # We still "pin" one color per y-series (so errorbars/lines/bands match).
    with use_economist_cycle():
        palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", None) or ECONOMIST_COLORS
    palette = list(palette)

    if mode == "scatter":
        for i, y in enumerate(y_cols_l):
            c = palette[i % len(palette)]
            ax.scatter(df[x_col], df[y], s=s, alpha=alpha, label=y, color=c)
        ax.set_xlabel(x_col)
        ax.legend(frameon=False)
        return fig, ax

    # binned mode
    res = _prepare_binned_stats(
        df,
        x_col,
        y_cols_l,
        cutoffs=cutoffs,
        n_bins=n_bins,
        include_inf=include_inf,
        label_style=label_style,
        right_closed=right_closed,
        bin_x=bin_x,
        err=err,
        band_k=band_k,
    )
    bdf = res.binned_df

    for i, y in enumerate(y_cols_l):
        c = palette[i % len(palette)]
        if show_points_in_binned:
            ax.scatter(df[x_col], df[y], s=s, alpha=min(alpha, 0.25), label=None, color=c)

        x_cent = bdf["x_center"].to_numpy(dtype=float)
        mu = bdf[f"{y}__mean"].to_numpy(dtype=float)
        lo = bdf[f"{y}__lo"].to_numpy(dtype=float)
        hi = bdf[f"{y}__hi"].to_numpy(dtype=float)

        if binned_style == "band":
            if connect_means:
                ax.plot(x_cent, mu, label=f"{y} (mean)", color=c)
            ax.scatter(
                x_cent,
                mu,
                s=30,
                zorder=3,
                label=(f"{y} (mean)" if not connect_means else None),
                color=c,
            )
            ax.fill_between(x_cent, lo, hi, alpha=band_alpha, color=c)
        else:
            # Categorical-style (like GAM factor plots): dot + vertical whiskers
            yerr = np.vstack([mu - lo, hi - mu])
            ax.errorbar(
                x_cent,
                mu,
                yerr=yerr,
                fmt="o",
                capsize=capsize,
                label=f"{y} (mean)",
                color=c,
                ecolor=c,
            )
            if connect_means:
                ax.plot(x_cent, mu, alpha=0.6, linewidth=1.0, label=None, color=c)

    ax.set_xlabel(x_col)
    ax.legend(frameon=False)

    # X tick labeling in binned mode
    if x_tick_labels != "none":
        ax.set_xticks(bdf["x_center"].to_numpy(dtype=float))

        if x_tick_labels == "label" and "bin_label" in bdf.columns:
            ax.set_xticklabels(bdf["bin_label"].tolist(), rotation=45, ha="right")
        else:
            # numeric labels at bin positions
            fmt = f"{{:.{int(x_tick_decimals)}f}}"
            ax.set_xticklabels([fmt.format(v) for v in bdf["x_center"].to_numpy(dtype=float)], rotation=0)

    return fig, ax


