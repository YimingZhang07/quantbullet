"""Binned means: weighted means of y per bin of x, drawn over count or weight bars.

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
    bar_mode: Literal["total", "stacked", "none"] = "total",
    bar_value: Literal["count", "weight"] = "count",
    wrap: int | None = None,
    panel_size: tuple[float, float] = (5.2, 3.5),
    # Axes
    y_scale: Literal["shared", "free"] | tuple[float, float] = "shared",
    bar_scale: Literal["shared", "free"] | float = "free",
    y_ticks: Literal["all", "outer"] = "all",
    bar_ticks: Literal["all", "outer"] = "all",
    y_titles: Literal["all", "outer"] = "all",
    bar_titles: Literal["all", "outer"] = "all",
    x_titles: Literal["all", "outer"] = "all",
    # Text
    labels: Mapping[str, str] | None = None,
    y_label: str = "Weighted mean",
    bar_label: str | None = None,
    title: str | None = None,
    y_format: str | None = None,
    # Look
    theme: PlotTheme = MINIMAL_THEME,
    style: BinnedMeansStyle = DEFAULT_BINNED_MEANS_STYLE,
) -> BinnedMeansPlot:
    """Aggregate then draw; see ``summarize_binned_means`` and ``draw_binned_means``.

    ``y_transform`` applies to bin-level means after weighting (e.g. SMM ->
    CPR); ``min_count`` hides curve values for bins with fewer rows while
    keeping their bars. ``bar_value='weight'`` sizes bars by the summed
    ``weight`` instead of rows. ``y_scale`` / ``bar_scale`` set each y
    axis's range, ``y_ticks`` / ``bar_ticks`` its tick labels and
    ``y_titles`` / ``bar_titles`` / ``x_titles`` the axis titles.

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
    return draw_binned_means(data, bar_mode=bar_mode, bar_value=bar_value, wrap=wrap, panel_size=panel_size,
                             y_scale=y_scale, bar_scale=bar_scale, y_ticks=y_ticks,
                             bar_ticks=bar_ticks, y_titles=y_titles, bar_titles=bar_titles,
                             x_titles=x_titles, labels=labels, y_label=y_label, bar_label=bar_label, title=title,
                             y_format=y_format, theme=theme, style=style)
