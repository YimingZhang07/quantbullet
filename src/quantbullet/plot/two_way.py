"""Draw two-way diagnostics: a heatmap of the two-way table above one slice panel per row (or column).

``draw_two_way_ae`` draws a ``TwoWayAE`` (actual / expected per bin, with row-count bars in each slice);
``draw_two_way_correction`` draws a ``TwoWayCorrection`` (the fitted multiplier D at the knots, with other
fits overlaid). The heatmap colours log values on a diverging scale centred at 1: blue below, red above.
Slices put a log-scale variable on a matplotlib log axis with ticks at 1-2-5 values.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, Literal, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

from .formatter import PlotFormatter, compact_number
from .panels import label_outer_panels
from .theme import MINIMAL_THEME, PlotTheme

if TYPE_CHECKING:
    from quantbullet.diagnostics.two_way import TwoWayAE, TwoWayCorrection

_COLORBAR_TICKS = (.5, .67, .8, .9, 1, 1.1, 1.25, 1.5, 2)


@dataclass(frozen=True)
class TwoWayStyle:
    """Visual settings of two-way diagnostic plots; fonts and text colours come from the theme."""

    low_color: str = "#184f95"
    mid_color: str = "#f0efec"
    high_color: str = "#b83232"
    min_color_range: float = 1.1          # the colour scale spans at least [1 / 1.1, 1.1]
    annotate: bool = True
    annotation_fontsize: float | None = None   # None: the theme's tick label size
    line_width: float = 1.6
    overlay_line_width: float = 1.3
    marker_size: float = 3.5
    capsize: float = 2.0
    bar_color: str = "#AEB8C2"
    bar_alpha: float = .27
    bar_width_ratio: float = .86
    reference_color: str = "#9A9A9A"


DEFAULT_TWO_WAY_STYLE = TwoWayStyle()


@dataclass(frozen=True, eq=False)
class TwoWayPlot:
    """The figure, its heatmap, the slice axes in panel order, and the result drawn."""

    fig: Figure
    heatmap_ax: Axes
    slice_axes: tuple
    result: object


# ---------------------------------------------------------------- labels and positions
def _number(value: float, _pos=None) -> str:
    return compact_number(value) if abs(value) >= 1e4 else f"{value:g}"


def _level_label(level, first: bool = False) -> str:
    if isinstance(level, pd.Interval):
        opening = "[" if first or level.closed == "both" else "("
        return f"{opening}{_number(level.left)}, {_number(level.right)}]"
    return _number(level) if isinstance(level, (int, float, np.number)) else str(level)


def _bin_edges(levels, scale: str, value_range) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Marker positions and bar edges of bins in data units, clipped to the observed range of the variable."""
    if levels and all(isinstance(level, pd.Interval) for level in levels):
        left = np.array([max(level.left, value_range[0]) for level in levels], dtype=float)
        right = np.array([min(level.right, value_range[1]) for level in levels], dtype=float)
        outside = left >= right   # bins beyond the data keep their own edges
        left[outside] = [levels[i].left for i in np.flatnonzero(outside)]
        right[outside] = [levels[i].right for i in np.flatnonzero(outside)]
    else:   # BinSpec.round values: half a step on each side
        centres = np.asarray(levels, dtype=float)
        if scale == "log":
            step = np.sqrt(np.min(centres[1:] / centres[:-1])) if len(centres) > 1 else 2.
            return centres, centres / step, centres * step
        half = np.min(np.diff(centres)) / 2 if len(centres) > 1 else .5
        return centres, centres - half, centres + half
    if scale == "log":
        left = np.where(left > 0, left, value_range[0])
        return np.sqrt(left * right), left, right
    return (left + right) / 2, left, right


def _shrink(left: np.ndarray, right: np.ndarray, scale: str, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Narrow bars around their centre, in the axis scale, so neighbours do not touch."""
    to_axis, from_axis = (np.log, np.exp) if scale == "log" else (np.asarray, np.asarray)
    a, b = to_axis(left), to_axis(right)
    centre, half = (a + b) / 2, (b - a) * ratio / 2
    return from_axis(centre - half), from_axis(centre + half)


def _value_axis(axes, scale: str) -> None:
    """The slices' shared horizontal axis in original units; a log scale gets ticks at 1-2-5 values."""
    for ax in axes:
        if scale == "log":
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(_number))


# ---------------------------------------------------------------- layout, heatmap, slices
def _check(slices, ci, n_cols, panel_size, heatmap_height, color_range, y_range):
    if slices not in ("x", "z"):
        raise ValueError("slices must be 'x' or 'z'")
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")
    if isinstance(n_cols, bool) or not isinstance(n_cols, (int, np.integer)) or n_cols < 1:
        raise ValueError("n_cols must be a positive integer")
    if len(panel_size) != 2 or not all(np.isfinite(v) and v > 0 for v in panel_size):
        raise ValueError("panel_size must contain two positive finite values")
    if not np.isfinite(heatmap_height) or heatmap_height <= 0:
        raise ValueError("heatmap_height must be positive")
    if color_range is not None and not (np.isscalar(color_range) and np.isfinite(color_range) and color_range > 1):
        raise ValueError("color_range must be a finite number above 1, or None")
    if y_range is not None and not (np.shape(y_range) == (2,) and np.isfinite(y_range).all()
                                    and y_range[0] < y_range[1]):
        raise ValueError("y_range must be (low, high) with finite low < high, or None")


def _layout(n_slices: int, n_cols: int, panel_size, heatmap_height: float):
    """A full-width heatmap above a grid of slices; the grid is returned padded with None."""
    columns = max(1, min(n_cols, n_slices))
    rows = int(np.ceil(n_slices / columns))
    fig = plt.figure(figsize=(panel_size[0] * columns, heatmap_height + panel_size[1] * rows), layout="constrained")
    spec = fig.add_gridspec(1 + rows, columns, height_ratios=[heatmap_height] + [panel_size[1]] * rows)
    heatmap = fig.add_subplot(spec[0, :])
    grid = np.full((rows, columns), None, dtype=object)
    for i in range(n_slices):
        first = grid.flat[0]
        grid.flat[i] = fig.add_subplot(spec[1 + i // columns, i % columns], sharex=first, sharey=first)
    return fig, heatmap, grid


def _heatmap(fig, ax, values, half_widths, fixed, row_labels, col_labels, row_title, col_title,
             colorbar_label, color_range, theme: PlotTheme, style: TwoWayStyle):
    """Log-coloured cells, x rows from the bottom; each cell shows its value and the CI half-width.

    Colours saturate at x color_range and / color_range; without it, at the largest value in the table.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(values > 0, np.log(values), np.nan)
    if color_range is not None:
        limit = np.log(color_range)
    else:
        finite = np.abs(logs[np.isfinite(logs)])
        limit = max(np.log(style.min_color_range), finite.max() if finite.size else 0.)
    cmap = LinearSegmentedColormap.from_list(
        "two_way", [style.low_color, style.mid_color, style.high_color]).with_extremes(bad="white")
    image = ax.imshow(np.ma.masked_invalid(logs), cmap=cmap, vmin=-limit, vmax=limit, origin="lower",
                      aspect="auto", interpolation="nearest")
    size = style.annotation_fontsize or theme.tick_labelsize
    if style.annotate:
        for (r, c), value in np.ndenumerate(values):
            if fixed[r, c]:
                ax.text(c, r, "1 · ref", ha="center", va="center", fontsize=size * .9, color=theme.muted_text_color)
            elif not np.isfinite(value):
                ax.text(c, r, "–", ha="center", va="center", fontsize=size, color=theme.muted_text_color)
            else:
                text = f"{value:.2f}"
                if np.isfinite(half_widths[r, c]):
                    text += f"\n±{half_widths[r, c]:.2f}"
                strong = abs(logs[r, c]) > .6 * limit
                ax.text(c, r, text, ha="center", va="center", fontsize=size,
                        color="white" if strong else theme.label_color)
    long_labels = max(map(len, col_labels)) > 6
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=30 if long_labels else 0,
                  ha="right" if long_labels else "center")
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0, labelsize=theme.tick_labelsize, colors=theme.tick_label_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(theme.secondary_spine_color)
    ax.set_xlabel(col_title, fontsize=theme.label_fontsize, color=theme.label_color)
    ax.set_ylabel(row_title, fontsize=theme.label_fontsize, color=theme.label_color)
    ticks = [t for t in _COLORBAR_TICKS if abs(np.log(t)) <= limit + 1e-12]
    bar = fig.colorbar(image, ax=ax, ticks=np.log(ticks), pad=.01, fraction=.04)
    bar.ax.set_yticklabels([f"{t:g}" for t in ticks])
    bar.ax.tick_params(labelsize=theme.tick_labelsize, colors=theme.tick_label_color)
    bar.set_label(colorbar_label, fontsize=theme.label_fontsize, color=theme.label_color)
    bar.outline.set_visible(False)


def _style_slice(ax, title: str, x_label: str, y_label: str, theme: PlotTheme, style: TwoWayStyle):
    PlotFormatter.apply_theme(ax, theme)
    ax.set_axisbelow(theme.grid_below)
    ax.axhline(1, color=style.reference_color, linewidth=.8, zorder=1)
    # Slice titles name a bin such as "current balance: (100K, 150K]"; label size keeps them inside the panel.
    ax.set_title(title, fontsize=theme.label_fontsize, pad=theme.title_pad, loc="left", color=theme.title_color)
    ax.set_xlabel(x_label, fontsize=theme.label_fontsize, color=theme.label_color)
    ax.set_ylabel(y_label, fontsize=theme.label_fontsize, color=theme.label_color)
    ax.tick_params(axis="x", labelbottom=True)
    ax.tick_params(axis="y", labelleft=True)


def _finish(fig, handles, title, theme: PlotTheme):
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 4),
               frameon=theme.legend_frameon, fontsize=theme.legend_fontsize)
    if title:
        size = theme.figure_title_fontsize or theme.title_fontsize + 2
        fig.suptitle(title, fontsize=size, fontweight=theme.title_fontweight, color=theme.title_color)


def _y_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = np.append(finite[np.isfinite(finite)], 1.)
    low, high = finite.min(), finite.max()
    pad = max(high - low, .05) * .08
    return low - pad, high + pad


# ---------------------------------------------------------------- A/E
def draw_two_way_ae(
    ae: TwoWayAE,
    *,
    slices: Literal["x", "z"] = "x",
    min_rows: int = 0,
    ci: float = .95,
    n_cols: int = 4,
    panel_size: tuple[float, float] = (3.2, 2.6),
    heatmap_height: float = 3.6,
    color_range: float | None = None,
    y_range: tuple[float, float] | None = None,
    title: str | None = None,
    theme: PlotTheme = MINIMAL_THEME,
    style: TwoWayStyle = DEFAULT_TWO_WAY_STYLE,
) -> TwoWayPlot:
    """Heatmap of A/E per (x bin, z bin) above one panel per x bin (``slices="z"``: per z bin).

    Each slice shows A/E with ``ci`` error bars from ``ae_se`` against the other variable, a line at 1 and
    background bars of row counts; all slices share the A/E and the count scales. Cells with fewer than
    ``min_rows`` rows are left blank in the heatmap and break the slice curves. ``color_range`` (colours
    saturate at x / divided by it) and ``y_range`` (the slices' A/E axis) fix the scales, e.g. so a figure
    before a correction and one after it compare directly; by default both follow the data.
    """
    _check(slices, ci, n_cols, panel_size, heatmap_height, color_range, y_range)
    if isinstance(min_rows, bool) or not isinstance(min_rows, (int, np.integer)) or min_rows < 0:
        raise ValueError("min_rows must be a nonnegative integer")
    q = NormalDist().inv_cdf(.5 + ci / 2)
    rows = ae.pivot("rows").fillna(0).to_numpy()
    sparse = rows < max(min_rows, 1)
    values = np.where(sparse, np.nan, ae.pivot("ae").to_numpy())
    half = np.where(sparse, np.nan, q * ae.pivot("ae_se").to_numpy())
    x_labels = [_level_label(level, i == 0) for i, level in enumerate(ae.x_levels)]
    z_labels = [_level_label(level, i == 0) for i, level in enumerate(ae.z_levels)]

    panel_levels, panel_labels, across_levels, across_scale, across_range, panel_name, across_name = (
        (ae.x_levels, x_labels, ae.z_levels, ae.z_scale, ae.z_range, ae.x, ae.z) if slices == "x" else
        (ae.z_levels, z_labels, ae.x_levels, ae.x_scale, ae.x_range, ae.z, ae.x))
    oriented = (lambda array: array) if slices == "x" else (lambda array: array.T)
    fig, heatmap, grid = _layout(len(panel_levels), n_cols, panel_size, heatmap_height)
    axes = [ax for ax in grid.flat if ax is not None]
    _heatmap(fig, heatmap, values, half, np.zeros(values.shape, dtype=bool), x_labels, z_labels,
             ae.label(ae.x), ae.label(ae.z), "A/E", color_range, theme, style)

    positions, left, right = _bin_edges(list(across_levels), across_scale, across_range)
    bar_left, bar_right = _shrink(left, right, across_scale, style.bar_width_ratio)
    bar_top = max(float(rows.max()), 1.) * 1.05
    color = theme.palette[0]
    twins = np.full(grid.shape, None, dtype=object)
    for i, ax in enumerate(axes):
        twin = twins.flat[i] = ax.twinx()
        ax.set_zorder(twin.get_zorder() + 1)
        ax.patch.set_visible(False)
        twin.bar(bar_left, oriented(rows)[i], width=bar_right - bar_left, align="edge", color=style.bar_color,
                 alpha=style.bar_alpha, edgecolor="none")
        twin.set_ylim(0, bar_top)
        twin.grid(False)
        twin.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        twin.yaxis.set_major_formatter(FuncFormatter(compact_number))
        twin.tick_params(axis="y", labelsize=theme.tick_labelsize, colors=theme.muted_text_color)
        twin.set_ylabel("Rows", fontsize=theme.label_fontsize, color=theme.muted_text_color)
        for spine in twin.spines.values():
            spine.set_visible(False)
        _style_slice(ax, f"{ae.label(panel_name)}: {panel_labels[i]}", ae.label(across_name), "A/E", theme, style)
        ax.errorbar(positions, oriented(values)[i], yerr=oriented(half)[i], color=color, linewidth=style.line_width,
                    marker="o", markersize=style.marker_size, capsize=style.capsize, elinewidth=.9, zorder=3)
    _value_axis(axes + [twin for twin in twins.flat if twin is not None], across_scale)
    if axes:
        axes[0].set_ylim(*(_y_limits(values) if y_range is None else y_range))
        axes[0].set_xlim(left.min(), right.max())
    label_outer_panels(grid, twins, y_titles=True, bar_titles=True)
    handles = [Line2D([], [], color=color, linewidth=style.line_width, marker="o", markersize=style.marker_size,
                      label=f"A/E (±{ci:.0%})"),
               Patch(facecolor=style.bar_color, alpha=style.bar_alpha, label="Rows (right axis)")]
    _finish(fig, handles, title, theme)
    return TwoWayPlot(fig, heatmap, tuple(axes), ae)


# ---------------------------------------------------------------- correction
def draw_two_way_correction(
    correction: TwoWayCorrection,
    *,
    overlays: Mapping[str, TwoWayCorrection] | None = None,
    slices: Literal["x", "z"] = "x",
    ci: float = .95,
    label: str = "fit",
    n_cols: int = 4,
    panel_size: tuple[float, float] = (3.2, 2.6),
    heatmap_height: float = 3.6,
    color_range: float | None = None,
    y_range: tuple[float, float] | None = None,
    title: str | None = None,
    theme: PlotTheme = MINIMAL_THEME,
    style: TwoWayStyle = DEFAULT_TWO_WAY_STYLE,
) -> TwoWayPlot:
    """Heatmap of the fitted D at the knots above one panel per x knot (``slices="z"``: per z knot).

    A reference line is all ones and gets no panel. Each slice shows the nodes with ``ci`` error bars joined by
    straight lines, which is D exactly along that knot, plus a line at 1 and a mark at the other axis's
    reference. ``overlays`` maps labels to other fits of the same x and z columns and scales (e.g. a subset),
    drawn along each slice; their knots may differ. ``color_range`` and ``y_range`` fix the heatmap colours and
    the slices' D axis, as in ``draw_two_way_ae``.
    """
    _check(slices, ci, n_cols, panel_size, heatmap_height, color_range, y_range)
    overlays = dict(overlays or {})
    for name, other in overlays.items():
        if (other.x, other.z, other.x_scale, other.z_scale) != (correction.x, correction.z, correction.x_scale,
                                                                 correction.z_scale):
            raise ValueError(f"overlay {name!r} must use the same x and z columns and scales")
    q = NormalDist().inv_cdf(.5 + ci / 2)

    if slices == "x":
        panel_knots, panel_name, panel_ref = correction.x_knots, correction.x, correction.x_reference
        across_knots, across_name, across_scale, across_ref = (correction.z_knots, correction.z,
                                                               correction.z_scale, correction.z_reference)
        nodes, se = correction.nodes, correction.se
    else:
        panel_knots, panel_name, panel_ref = correction.z_knots, correction.z, correction.z_reference
        across_knots, across_name, across_scale, across_ref = (correction.x_knots, correction.x,
                                                               correction.x_scale, correction.x_reference)
        nodes, se = correction.nodes.T, correction.se.T
    shown = [i for i, knot in enumerate(panel_knots) if panel_ref is None or not np.isclose(knot, panel_ref)]
    fig, heatmap, grid = _layout(len(shown), n_cols, panel_size, heatmap_height)
    axes = [ax for ax in grid.flat if ax is not None]
    _heatmap(fig, heatmap, correction.nodes, q * correction.se, ~correction.free,
             [_number(k) for k in correction.x_knots], [_number(k) for k in correction.z_knots],
             correction.label(correction.x), correction.label(correction.z), "multiplier D", color_range, theme, style)

    spacing = np.geomspace if across_scale == "log" else np.linspace
    along = spacing(across_knots[0], across_knots[-1], 200)
    colors = theme.palette
    curves = [nodes[shown]]
    for ax, i in zip(axes, shown):
        _style_slice(ax, f"{correction.label(panel_name)} = {_number(panel_knots[i])}",
                     correction.label(across_name), "multiplier D", theme, style)
        if across_ref is not None:
            ax.axvline(across_ref, color=theme.grid_color, linewidth=.8, zorder=1)
        ax.errorbar(across_knots, nodes[i], yerr=np.where(se[i] > 0, q * se[i], np.nan), color=colors[0],
                    linewidth=style.line_width, marker="o", markersize=style.marker_size, capsize=style.capsize,
                    elinewidth=.9, zorder=3)
        for k, other in enumerate(overlays.values(), start=1):
            fixed = np.full(len(along), panel_knots[i])
            curve = other.surface(fixed, along) if slices == "x" else other.surface(along, fixed)
            curves.append(curve)
            ax.plot(along, curve, color=colors[k % len(colors)], linewidth=style.overlay_line_width, zorder=2)
    _value_axis(axes, across_scale)
    if axes:
        drawn = np.concatenate([np.ravel(c) for c in curves])
        axes[0].set_ylim(*(_y_limits(drawn) if y_range is None else y_range))
    label_outer_panels(grid, y_titles=True)
    handles = [Line2D([], [], color=colors[0], linewidth=style.line_width, marker="o",
                      markersize=style.marker_size, label=f"{label} (±{ci:.0%})")]
    handles += [Line2D([], [], color=colors[k % len(colors)], linewidth=style.overlay_line_width, label=name)
                for k, name in enumerate(overlays, start=1)]
    _finish(fig, handles, title, theme)
    return TwoWayPlot(fig, heatmap, tuple(axes), correction)
