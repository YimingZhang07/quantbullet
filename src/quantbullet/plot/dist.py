from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from pandas.api.types import is_numeric_dtype

from quantbullet.plot.utils import get_grid_fig_axes, close_unused_axes


_HIST_COLOR = "#4878A8"
_KDE_COLOR = "#2B4D6F"
_MEAN_COLOR = "#D64545"
_MEDIAN_COLOR = "#E8943A"
_BAR_COLOR = "#4878A8"
_STAT_BOX_PROPS = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CCCCCC", alpha=0.88)


def _annotate_numeric_stats(ax, series: pd.Series, decimals: int = 2) -> None:
    fmt = f".{decimals}f"
    lines = [
        f"N      = {len(series):,}",
        f"Mean   = {series.mean():{fmt}}",
        f"Median = {series.median():{fmt}}",
        f"Std    = {series.std():{fmt}}",
        f"Min    = {series.min():{fmt}}",
        f"Max    = {series.max():{fmt}}",
    ]
    ax.text(
        0.97, 0.95, "\n".join(lines),
        transform=ax.transAxes, fontsize=7, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="right",
        bbox=_STAT_BOX_PROPS,
    )


def _plot_numeric(ax, series: pd.Series, col: str, bins: int, show_stats: bool = True, stat_decimals: int = 2) -> None:
    sns.histplot(
        series, bins=bins, kde=True, ax=ax,
        color=_HIST_COLOR, edgecolor="white", linewidth=0.4, alpha=0.75,
        line_kws=dict(color=_KDE_COLOR, linewidth=1.2),
        stat="density",
    )
    mean_val = series.mean()
    median_val = series.median()
    ax.axvline(mean_val, color=_MEAN_COLOR, linestyle="--", linewidth=1.0, label=f"Mean ({mean_val:.3g})")
    ax.axvline(median_val, color=_MEDIAN_COLOR, linestyle=":", linewidth=1.0, label=f"Median ({median_val:.3g})")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    ax.set_title(col, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if show_stats:
        _annotate_numeric_stats(ax, series, decimals=stat_decimals)


def _plot_categorical(ax, series: pd.Series, col: str, max_categories: int) -> None:
    counts = series.astype(str).value_counts()
    if len(counts) > max_categories:
        top = counts.head(max_categories - 1)
        other_count = counts.iloc[max_categories - 1:].sum()
        counts = pd.concat([top, pd.Series({"Other": other_count})])

    counts = counts.sort_values(ascending=True)

    bars = ax.barh(
        y=range(len(counts)), width=counts.values,
        color=_BAR_COLOR, edgecolor="white", linewidth=0.4, alpha=0.85,
        height=0.7,
    )
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=8)

    max_val = counts.max()
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", ha="left", fontsize=7, color="#333333",
        )

    ax.set_xlim(0, max_val * 1.15)
    n_unique = series.nunique()
    ax.set_title(f"{col}  ({n_unique} unique)", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("")


def plot_distributions(
    df: pd.DataFrame,
    columns: list[str],
    *,
    n_cols: int = 3,
    width: float = 4.5,
    height: float = 3.5,
    bins: int = 30,
    max_categories: int = 20,
    cat_threshold: int = 15,
    sample: int | float | None = None,
    suptitle: str | None = None,
    stat_decimals: int | None = 2,
) -> Tuple[Figure, np.ndarray]:
    """Plot distribution panels for a list of DataFrame columns.

    Numeric columns render as histograms with KDE overlay and summary
    statistics.  Categorical columns (or numeric columns with fewer than
    *cat_threshold* unique values) render as horizontal bar charts.

    Parameters
    ----------
    df : DataFrame
    columns : list[str]
        Column names to plot.
    n_cols : int
        Grid columns.
    width, height : float
        Size of each subplot.
    bins : int
        Histogram bins for numeric columns.
    max_categories : int
        Show at most this many bars for categorical columns; the rest
        are bucketed as "Other".
    cat_threshold : int
        Numeric columns with <= this many unique values are treated
        as categorical.
    sample : int | float | None
        Down-sample before plotting (int = n rows, float = fraction).
    suptitle : str | None
        Figure super-title.
    stat_decimals : int | None
        Decimal places for the stats annotation on numeric panels.
        ``None`` hides the annotation entirely. Default 2.

    Returns
    -------
    fig, axes
    """
    columns = [c for c in columns if c in df.columns]
    if not columns:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.text(0.5, 0.5, "No columns to plot", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig, np.array([ax])

    if sample is not None:
        if isinstance(sample, float):
            df = df.sample(frac=sample, random_state=42)
        else:
            df = df.sample(n=min(sample, len(df)), random_state=42)

    with sns.axes_style("whitegrid"):
        fig, axes = get_grid_fig_axes(n_charts=len(columns), n_cols=n_cols, width=width, height=height)

        for i, col in enumerate(columns):
            ax = axes[i]
            series = df[col].dropna()

            is_cat = not is_numeric_dtype(series) or series.nunique() <= cat_threshold
            if is_cat:
                _plot_categorical(ax, series, col, max_categories)
            else:
                _plot_numeric(ax, series, col, bins, show_stats=stat_decimals is not None, stat_decimals=stat_decimals or 2)

        close_unused_axes(axes)

        if suptitle:
            fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.98] if suptitle else None)

    return fig, axes
