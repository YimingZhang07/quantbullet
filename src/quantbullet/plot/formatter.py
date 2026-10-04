from __future__ import annotations

import numpy as np
from matplotlib import ticker as mticker

from quantbullet.plot.theme import PlotTheme


def compact_number(value, _pos=None) -> str:
    """Tick label with a K/M/B suffix: 450000 -> '450K', 1.2e6 -> '1.2M'."""
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= divisor:
            return mticker.Formatter.fix_minus(f"{value / divisor:g}{suffix}")
    return mticker.Formatter.fix_minus(f"{value:g}")


class StepPercentFormatter(mticker.Formatter):
    """Percent labels for fractions, with as many decimals as the tick step needs.

    Steps of 1% print '4%', steps of 0.5% print '4.5%', so every label on an
    axis has the same number of decimals.
    """

    _decimals = 1

    def set_locs(self, locs):
        super().set_locs(locs)
        ticks = np.unique(np.asarray(locs, dtype=float))
        if len(ticks) > 1:
            step = float(np.min(np.diff(ticks))) * 100
            self._decimals = next((d for d in range(3) if abs(round(step, d) - step) <= 1e-6 * max(step, 1)), 3)

    def __call__(self, x, pos=None):
        return self.fix_minus(f"{x * 100:.{self._decimals}f}%")


class PlotFormatter:
    """Stateless toolkit for styling and formatting matplotlib axes."""

    # -- theme application ----------------------------------------------------

    @staticmethod
    def apply_theme(ax, theme: PlotTheme) -> None:
        """Apply visual chrome from a :class:`PlotTheme` to a matplotlib ``Axes``.

        Configures spines, grid, background, and tick appearance.
        Does **not** set any text content — use :meth:`set_title` /
        :meth:`set_labels` for that.
        """
        t = theme

        if t.facecolor is not None:
            ax.set_facecolor(t.facecolor)

        # Spines
        ax.spines["top"].set_visible(not t.hide_top_spine)
        ax.spines["right"].set_visible(not t.hide_right_spine)
        ax.spines["left"].set_visible(not t.hide_left_spine)
        ax.spines["bottom"].set_visible(not t.hide_bottom_spine)
        for sp in ax.spines.values():
            sp.set_edgecolor(t.spine_color)
            sp.set_linewidth(t.spine_linewidth)

        # Grid
        if t.grid:
            ax.grid(
                True,
                axis=t.grid_axis,
                color=t.grid_color,
                linestyle=t.grid_linestyle,
                linewidth=t.grid_linewidth,
            )
            if t.grid_axis == "y":
                ax.xaxis.grid(False)
            elif t.grid_axis == "x":
                ax.yaxis.grid(False)
            if t.grid_below:
                ax.set_axisbelow(True)
        else:
            ax.grid(False)

        # Ticks
        ax.tick_params(
            axis="both",
            labelsize=t.tick_labelsize,
            colors=t.tick_color,
            labelcolor=t.tick_label_color,
            direction=t.tick_direction,
            length=t.tick_length,
            width=t.tick_width,
        )

    # -- text / labels --------------------------------------------------------

    @staticmethod
    def set_title(ax, title: str) -> None:
        """Set the axes title."""
        ax.set_title(title)

    @staticmethod
    def set_labels(
        ax,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> None:
        """Set axis labels."""
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
