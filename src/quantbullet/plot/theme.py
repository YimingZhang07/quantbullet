from __future__ import annotations

from dataclasses import dataclass, replace  # noqa: F401 – re-export replace for users


@dataclass(frozen=True)
class PlotTheme:
    """Immutable bundle of visual parameters for matplotlib plots.

    Usage
    -----
    >>> from quantbullet.plot import ECONOMIST_THEME
    >>> ECONOMIST_THEME.apply(ax, title="My Chart", xlabel="X", ylabel="Y")

    Derive a custom theme with ``dataclasses.replace``::

        from dataclasses import replace
        big = replace(ECONOMIST_THEME, title_fontsize=16, tick_labelsize=12)
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

    # -- title ----------------------------------------------------------------
    title_fontsize: float = 12
    title_fontweight: str = "bold"
    title_pad: float = 10
    title_loc: str = "center"
    title_color: str = "#1A1A1A"

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

    def apply(
        self,
        ax,
        *,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> None:
        """Apply this theme to a matplotlib ``Axes``."""

        # Background
        if self.facecolor is not None:
            ax.set_facecolor(self.facecolor)

        # Spines
        ax.spines["top"].set_visible(not self.hide_top_spine)
        ax.spines["right"].set_visible(not self.hide_right_spine)
        ax.spines["left"].set_visible(not self.hide_left_spine)
        ax.spines["bottom"].set_visible(not self.hide_bottom_spine)
        for sp in ax.spines.values():
            sp.set_edgecolor(self.spine_color)
            sp.set_linewidth(self.spine_linewidth)

        # Grid
        if self.grid:
            ax.grid(
                True,
                axis=self.grid_axis,
                color=self.grid_color,
                linestyle=self.grid_linestyle,
                linewidth=self.grid_linewidth,
            )
            if self.grid_axis == "y":
                ax.xaxis.grid(False)
            elif self.grid_axis == "x":
                ax.yaxis.grid(False)
            if self.grid_below:
                ax.set_axisbelow(True)
        else:
            ax.grid(False)

        # Title
        if title:
            ax.set_title(
                title,
                fontsize=self.title_fontsize,
                fontweight=self.title_fontweight,
                pad=self.title_pad,
                loc=self.title_loc,
                color=self.title_color,
            )

        # Axis labels
        if xlabel is not None:
            ax.set_xlabel(
                xlabel,
                fontsize=self.label_fontsize,
                fontweight=self.label_fontweight,
                color=self.label_color,
            )
        if ylabel is not None:
            ax.set_ylabel(
                ylabel,
                fontsize=self.label_fontsize,
                fontweight=self.label_fontweight,
                color=self.label_color,
            )

        # Ticks
        ax.tick_params(
            axis="both",
            labelsize=self.tick_labelsize,
            colors=self.tick_color,
            labelcolor=self.tick_label_color,
            direction=self.tick_direction,
            length=self.tick_length,
            width=self.tick_width,
        )


# ---------------------------------------------------------------------------
# Preset themes
# ---------------------------------------------------------------------------

ECONOMIST_THEME = PlotTheme()

PRESENTATION_THEME = PlotTheme(
    title_fontsize=16,
    title_fontweight="bold",
    label_fontsize=13,
    label_fontweight="bold",
    tick_labelsize=11,
    tick_length=5.0,
    spine_linewidth=1.0,
    grid_linewidth=0.7,
)
