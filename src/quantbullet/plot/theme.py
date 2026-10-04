from __future__ import annotations

from dataclasses import dataclass, replace  # noqa: F401 – re-export replace for users

from .cycles import ECONOMIST_COLORS


@dataclass(frozen=True)
class PlotTheme:
    """Immutable bundle of visual parameters for matplotlib plots.

    Pure configuration — use :class:`PlotFormatter` to apply a theme
    to an axes.

    Usage
    -----
    >>> from quantbullet.plot import PlotFormatter, MINIMAL_THEME
    >>> fmt = PlotFormatter(theme=MINIMAL_THEME)
    >>> fmt.apply_theme(ax, title="My Chart", xlabel="X", ylabel="Y")

    Derive a custom theme with ``dataclasses.replace``::

        from dataclasses import replace
        big = replace(MINIMAL_THEME, title_fontsize=16, tick_labelsize=12)
    """

    # -- spines ---------------------------------------------------------------
    hide_top_spine: bool = False
    hide_right_spine: bool = False
    hide_left_spine: bool = False
    hide_bottom_spine: bool = False
    spine_color: str = "#333333"
    spine_linewidth: float = 0.8

    # -- grid -----------------------------------------------------------------
    grid: bool = True
    grid_axis: str = "both"
    grid_color: str = "#E0E0E0"
    grid_linestyle: str = "-"
    grid_linewidth: float = 0.5
    grid_below: bool = True

    # -- background -----------------------------------------------------------
    facecolor: str | None = None

    # -- shared colors --------------------------------------------------------
    palette: tuple[str, ...] = tuple(ECONOMIST_COLORS)
    muted_text_color: str = "#777777"
    secondary_spine_color: str = "#BBBBBB"

    # -- title ----------------------------------------------------------------
    title_fontsize: float = 12
    title_fontweight: str = "bold"
    title_pad: float = 10
    title_loc: str = "center"
    title_color: str = "#1A1A1A"
    figure_title_fontsize: float | None = None

    # -- axis labels ----------------------------------------------------------
    label_fontsize: float = 10
    label_fontweight: str = "normal"
    label_color: str = "#333333"

    # -- ticks ----------------------------------------------------------------
    tick_labelsize: float = 9
    tick_color: str = "#333333"
    tick_label_color: str = "#333333"
    tick_direction: str = "out"
    tick_length: float = 4.0
    tick_width: float = 0.6

    # -- legend ---------------------------------------------------------------
    legend_frameon: bool = False
    legend_fontsize: float = 9


# ---------------------------------------------------------------------------
# Preset themes
# ---------------------------------------------------------------------------

MINIMAL_THEME = PlotTheme()

# Print scale for figures placed 1:1 in a PDF whose body text is 9pt.
PRINT_THEME = PlotTheme(
    title_fontsize=8.5, title_pad=3, label_fontsize=8, tick_labelsize=7,
    tick_length=2.5, legend_fontsize=7.5,
)

# The same scale for plots that read matplotlib rcParams instead of a theme.
PRINT_RC = {
    "font.size": PRINT_THEME.label_fontsize,
    "axes.titlesize": PRINT_THEME.title_fontsize,
    "axes.titlepad": PRINT_THEME.title_pad,
    "axes.labelsize": PRINT_THEME.label_fontsize,
    "xtick.labelsize": PRINT_THEME.tick_labelsize,
    "ytick.labelsize": PRINT_THEME.tick_labelsize,
    "xtick.major.size": PRINT_THEME.tick_length,
    "ytick.major.size": PRINT_THEME.tick_length,
    "legend.fontsize": PRINT_THEME.legend_fontsize,
    "figure.titlesize": PRINT_THEME.title_fontsize + 0.5,
    "lines.linewidth": 1.3,
    "lines.markersize": 3,
}
