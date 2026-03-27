from __future__ import annotations

from quantbullet.plot.theme import PlotTheme


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
