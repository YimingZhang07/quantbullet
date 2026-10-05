"""Binned means: weighted means of y per bin of x, drawn over count bars.

The three verbs share one summary object, ``BinnedMeans``:

- ``summarize_binned_means`` aggregates raw rows once (``summary``);
- ``draw_binned_means`` draws a summary, or any subset of it (``draw``);
- ``plot_binned_means`` does both in one call.

``BinSpec`` (``binning``) says how each dimension is cut into bins.
"""
from __future__ import annotations

from typing import Callable, Literal, Mapping, Sequence

from ..theme import MINIMAL_THEME, PlotTheme
from .binning import BinSpec
from .draw import (DEFAULT_BINNED_MEANS_STYLE, PRINT_BINNED_MEANS_STYLE, BinnedMeansPlot, BinnedMeansStyle,
                   draw_binned_means)
from .summary import BinnedMeans, summarize_binned_means

__all__ = [
    "BinSpec", "BinnedMeans", "BinnedMeansPlot", "BinnedMeansStyle",
    "DEFAULT_BINNED_MEANS_STYLE", "PRINT_BINNED_MEANS_STYLE",
    "draw_binned_means", "plot_binned_means", "summarize_binned_means",
]


def plot_binned_means(
    df,
    *,
    # Data
    x: str,
    y: str | Sequence[str],
    group: str | None = None,
    row: str | None = None,
    col: str | None = None,
    # Aggregation
    weight: str | None = None,
    bins: Mapping[str, BinSpec] | None = None,
    y_transform: Callable | None = None,
    min_count: int = 0,
    # Layout
    count_mode: Literal["total", "stacked", "none"] = "total",
    wrap: int | None = None,
    panel_size: tuple[float, float] = (5.2, 3.5),
    # Axes
    y_scale: Literal["shared", "free"] | tuple[float, float] = "shared",
    count_scale: Literal["shared", "free"] | float = "free",
    y_ticks: Literal["all", "outer"] = "all",
    count_ticks: Literal["all", "outer"] = "all",
    y_titles: Literal["all", "outer"] = "all",
    count_titles: Literal["all", "outer"] = "all",
    x_titles: Literal["all", "outer"] = "all",
    # Text
    labels: Mapping[str, str] | None = None,
    y_label: str = "Weighted mean",
    title: str | None = None,
    y_format: str | None = None,
    # Look
    theme: PlotTheme = MINIMAL_THEME,
    style: BinnedMeansStyle = DEFAULT_BINNED_MEANS_STYLE,
) -> BinnedMeansPlot:
    """Aggregate then draw; see ``summarize_binned_means`` and ``draw_binned_means``.

    ``y_transform`` applies to bin-level means after weighting (e.g. SMM ->
    CPR); ``min_count`` hides curve values for bins with fewer rows while
    keeping their count bars. ``y_scale`` / ``count_scale`` set each y
    axis's range, ``y_ticks`` / ``count_ticks`` its tick labels and
    ``y_titles`` / ``count_titles`` / ``x_titles`` the axis titles.

    Example::

        result = plot_binned_means(
            df, x="incentive", y=["historical_cpr", "model_cpr"],
            weight="upb", bins={"incentive": BinSpec.step(0.25)},
            col="vintage", wrap=3, y_format=".0%",
        )
    """
    data = summarize_binned_means(df, x=x, y=y, weight=weight, group=group,
                                  row=row, col=col, bins=bins)
    if y_transform is not None:
        data = data.transform_means(y_transform)
    if min_count:
        data = data.mask_sparse(min_count)
    return draw_binned_means(data, count_mode=count_mode, wrap=wrap, panel_size=panel_size,
                             y_scale=y_scale, count_scale=count_scale, y_ticks=y_ticks,
                             count_ticks=count_ticks, y_titles=y_titles, count_titles=count_titles,
                             x_titles=x_titles, labels=labels, y_label=y_label, title=title,
                             y_format=y_format, theme=theme, style=style)
