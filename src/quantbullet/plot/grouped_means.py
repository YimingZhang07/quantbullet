"""Weighted-mean curves with count overlays and optional grouping/facets."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .formatter import PlotFormatter
from .grouped_data import BinSpec, GroupedMeansData, summarize_grouped_means
from .theme import MINIMAL_THEME, PlotTheme


@dataclass
class GroupedMeansPlot:
    """Figure, 2-D primary/secondary axes, and the exact aggregation plotted.

    Unused wrapped slots are hidden with ``None`` in ``count_axes``. With
    ``count_mode='none'`` all secondary axes entries are ``None``.
    """

    fig: Figure
    axes: np.ndarray
    count_axes: np.ndarray
    data: GroupedMeansData

    @property
    def summary(self) -> pd.DataFrame:
        return self.data.summary

    @property
    def bin_info(self) -> dict[str, dict]:
        return self.data.bin_info


@dataclass(frozen=True)
class GroupedMeansStyle:
    """Visual settings specific to weighted-mean curves and count bars."""

    metric_linestyles: tuple[str, ...] = ("-", "--", "-.", ":")
    line_width: float = 1.8
    legend_line_width: float = 2.0
    marker: str | None = "o"
    marker_size: float = 3.0
    count_color: str = "#AEB8C2"
    count_edgecolor: str = "none"
    total_count_alpha: float = 0.27
    stacked_count_alpha: float = 0.18
    bar_width_ratio: float = 0.86
    categorical_tick_rotation: float = 30.0


DEFAULT_GROUPED_MEANS_STYLE = GroupedMeansStyle()


def _level_label(value, first=False):
    if isinstance(value, pd.Interval):
        return f"{'[' if first or value.closed == 'both' else '('}{value.left:g}, {value.right:g}]"
    return str(value)


def draw_grouped_means(
    data: GroupedMeansData,
    *,
    count_mode: Literal["total", "stacked", "none"] = "total",
    wrap: int | None = None,
    panel_size: tuple[float, float] = (5.2, 3.5),
    share_y: bool = True,
    share_count_y: bool = False,
    theme: PlotTheme = MINIMAL_THEME,
    style: GroupedMeansStyle = DEFAULT_GROUPED_MEANS_STYLE,
    labels: Mapping[str, str] | None = None,
    title: str | None = None,
    ylabel: str = "Weighted mean",
    y_format: str | None = None,
) -> GroupedMeansPlot:
    """Render reusable statistics; no raw-data aggregation happens here.

    ``labels`` maps source column names to display labels. ``y_format`` is a
    Python format spec, e.g. '.0%' for proportions. ``panel_size`` is in inches
    per subplot. Numeric/binned x uses numeric positions, categories are
    equally spaced. Counts are rows, never duplicated across y metrics.

    With group, colors identify groups and line styles identify metrics;
    otherwise colors identify metrics. ``stacked`` requires group. Axes use
    a common x scale, and primary y scales are shared by default. Count axes
    are independent unless ``share_count_y=True``. Missing means break lines.
    """
    if count_mode not in {"total", "stacked", "none"}:
        raise ValueError("count_mode must be total, stacked, or none")
    if count_mode == "stacked" and "group" not in data.dimensions:
        raise ValueError("stacked counts require group")
    if wrap is not None and (isinstance(wrap, bool) or not isinstance(wrap, int) or wrap < 1 or "col" not in data.dimensions or "row" in data.dimensions):
        raise ValueError("wrap must be a positive integer and requires col without row")
    if len(panel_size) != 2 or not all(np.isfinite(v) and v > 0 for v in panel_size):
        raise ValueError("panel_size must contain two positive finite values")
    if not theme.palette:
        raise ValueError("theme.palette must contain at least one color")
    if not style.metric_linestyles:
        raise ValueError("style.metric_linestyles must contain at least one line style")
    labels = dict(labels or {})
    dim = data.dimensions
    rows = data.levels.get("row", (None,)) or (None,)
    cols = data.levels.get("col", (None,)) or (None,)
    panels = list(product(rows, cols))
    ncols = min(wrap, len(cols)) if wrap else len(cols)
    nrows = int(np.ceil(len(panels) / ncols)) if wrap else len(rows)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=share_y,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows), layout="constrained")
    count_axes = np.full(axes.shape, None, dtype=object)
    palette = theme.palette
    styles = style.metric_linestyles
    groups = data.levels.get("group", (None,))
    xlevels = list(data.levels["x"])
    xpos, widths = data.x_positions, data.x_widths * style.bar_width_ratio
    for index, (rv, cv) in enumerate(panels):
        ax = axes.flat[index]
        panel = data.summary
        for role, value in (("row", rv), ("col", cv)):
            if role in dim:
                panel = panel.loc[panel[role] == value]
        PlotFormatter.apply_theme(ax, theme)
        ax.set_axisbelow(theme.grid_below)
        twin = None
        if count_mode != "none":
            twin = ax.twinx()
            count_axes.flat[index] = twin
            # Draw the primary axes last, with its background transparent.
            ax.set_zorder(twin.get_zorder() + 1)
            ax.patch.set_visible(False)
            twin.set_facecolor(ax.get_facecolor())
            twin.grid(False)
            twin.set_ylabel("Count", fontsize=theme.label_fontsize, color=theme.muted_text_color)
            twin.tick_params(axis="y", labelsize=theme.tick_labelsize, colors=theme.muted_text_color)
            twin.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            for spine in twin.spines.values():
                spine.set_visible(False)
            twin.spines["right"].set_visible(True)
            twin.spines["right"].set_color(theme.secondary_spine_color)
            if count_mode == "total":
                counts = panel.groupby("x", observed=True)["count"].sum().reindex(xlevels, fill_value=0)
                twin.bar(xpos, counts, width=widths, color=style.count_color,
                         alpha=style.total_count_alpha, edgecolor=style.count_edgecolor)
            else:
                bottom = np.zeros(len(xlevels))
                for gi, value in enumerate(groups):
                    counts = panel.loc[panel["group"] == value].set_index("x")["count"].reindex(xlevels, fill_value=0).to_numpy()
                    twin.bar(xpos, counts, bottom=bottom, width=widths,
                             color=palette[gi % len(palette)], alpha=style.stacked_count_alpha,
                             edgecolor=style.count_edgecolor)
                    bottom += counts
            twin.set_ylim(bottom=0)
            if panel.empty or not panel["count"].sum():
                twin.set_ylim(0, 1)

        for gi, value in enumerate(groups):
            subset = panel.loc[panel["group"] == value] if "group" in dim else panel
            subset = subset.set_index("x").reindex(xlevels)
            for mi, metric in enumerate(data.metrics):
                color = palette[(gi if "group" in dim else mi) % len(palette)]
                linestyle = styles[mi % len(styles)] if "group" in dim else styles[0]
                ax.plot(xpos, subset[f"{metric}__mean"].to_numpy(dtype=float), color=color,
                        linestyle=linestyle, linewidth=style.line_width,
                        marker=style.marker, markersize=style.marker_size)
        if panel.empty or not panel["count"].sum():
            ax.text(0.5, 0.5, "No observations", transform=ax.transAxes, ha="center",
                    color=theme.muted_text_color)

        caption = []
        for role, value in (("row", rv), ("col", cv)):
            if role in dim:
                caption.append(f"{labels.get(dim[role], dim[role])}: {_level_label(value, value == data.levels[role][0] if data.levels[role] else False)}")
        ax.set_title(" | ".join(caption), fontsize=theme.title_fontsize, fontweight=theme.title_fontweight,
                     pad=theme.title_pad, loc=theme.title_loc, color=theme.title_color)
        ax.set_xlabel(labels.get(dim["x"], dim["x"]), fontsize=theme.label_fontsize,
                      fontweight=theme.label_fontweight, color=theme.label_color)
        ax.set_ylabel(ylabel, fontsize=theme.label_fontsize, fontweight=theme.label_fontweight, color=theme.label_color)
        if y_format is not None:
            ax.yaxis.set_major_formatter(lambda value, _: format(value, y_format))
        if xlevels and data.bin_info[dim["x"]].get("categorical", False):
            ax.set_xticks(xpos, [str(v) for v in xlevels],
                          rotation=style.categorical_tick_rotation, ha="right")
        # Wrapped grids can have a hidden last-row slot. Keep every visible
        # panel's x ticks readable even when its shared-axis sibling is hidden.
        ax.tick_params(axis="x", labelbottom=True)
        if len(xpos):
            ax.set_xlim(np.min(xpos - widths / 2) - widths.min() * 0.12,
                        np.max(xpos + widths / 2) + widths.min() * 0.12)

    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    if share_count_y:
        twins = [ax for ax in count_axes.flat if ax is not None]
        maximum = max((ax.get_ylim()[1] for ax in twins), default=1)
        for ax in twins:
            ax.set_ylim(0, maximum)

    handles = []
    if "group" in dim:
        for gi, value in enumerate(groups):
            handles.append(Line2D([], [], color=palette[gi % len(palette)],
                                  label=f"{labels.get(dim['group'], dim['group'])}: {_level_label(value, gi == 0)}",
                                  linewidth=style.legend_line_width))
        for mi, metric in enumerate(data.metrics):
            handles.append(Line2D([], [], color=theme.label_color,
                                  linestyle=styles[mi % len(styles)], label=labels.get(metric, metric),
                                  linewidth=style.legend_line_width,
                                  marker=style.marker, markersize=style.marker_size))
    else:
        handles = [Line2D([], [], color=palette[mi % len(palette)],
                          linestyle=styles[0], label=labels.get(metric, metric),
                          linewidth=style.legend_line_width,
                          marker=style.marker, markersize=style.marker_size)
                   for mi, metric in enumerate(data.metrics)]
    if count_mode == "total":
        handles.append(Patch(facecolor=style.count_color, alpha=style.total_count_alpha,
                             edgecolor=style.count_edgecolor, label="Count (right axis)"))
    elif count_mode == "stacked":
        for gi, value in enumerate(groups):
            handles.append(Patch(facecolor=palette[gi % len(palette)],
                                 alpha=style.stacked_count_alpha, edgecolor=style.count_edgecolor,
                                 label=f"Count: {_level_label(value, gi == 0)} (right axis)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 4),
               frameon=theme.legend_frameon, fontsize=theme.legend_fontsize)
    if title:
        title_size = theme.figure_title_fontsize
        fig.suptitle(title, fontsize=theme.title_fontsize + 2 if title_size is None else title_size,
                     fontweight=theme.title_fontweight, color=theme.title_color)
    return GroupedMeansPlot(fig, axes, count_axes, data)


def plot_grouped_means(
    df,
    *,
    x: str,
    y: str | Sequence[str],
    weight: str | None = None,
    group: str | None = None,
    row: str | None = None,
    col: str | None = None,
    bins: Mapping[str, BinSpec] | None = None,
    count_mode: Literal["total", "stacked", "none"] = "total",
    wrap: int | None = None,
    panel_size: tuple[float, float] = (5.2, 3.5),
    share_y: bool = True,
    share_count_y: bool = False,
    theme: PlotTheme = MINIMAL_THEME,
    style: GroupedMeansStyle = DEFAULT_GROUPED_MEANS_STYLE,
    labels: Mapping[str, str] | None = None,
    title: str | None = None,
    ylabel: str = "Weighted mean",
    y_format: str | None = None,
) -> GroupedMeansPlot:
    """Aggregate then plot; see summarize_grouped_means/draw_grouped_means.

    Example::

        result = plot_grouped_means(
            df, x="incentive", y=["historical_cpr", "model_cpr"],
            weight="upb", bins={"incentive": BinSpec.step(0.25)},
            col="vintage", wrap=3, y_format=".0%",
        )
    """
    data = summarize_grouped_means(df, x=x, y=y, weight=weight, group=group,
                                   row=row, col=col, bins=bins)
    return draw_grouped_means(data, count_mode=count_mode, wrap=wrap, panel_size=panel_size,
                              share_y=share_y, share_count_y=share_count_y, theme=theme, style=style,
                              labels=labels, title=title, ylabel=ylabel, y_format=y_format)
