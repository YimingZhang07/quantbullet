"""Panel grids: sets of panels drawn in any subset at any size, and their outer labels."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def panel_grid(n_panels: int, n_cols: int, panel_size: tuple[float, float]) -> tuple[Figure, np.ndarray]:
    """Constrained-layout grid for a ``PanelSet`` render; slots past ``n_panels`` are hidden."""
    n_rows = -(-n_panels // n_cols)
    fig, grid = plt.subplots(n_rows, n_cols, figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
                             squeeze=False, layout="constrained")
    for ax in grid.flat[n_panels:]:
        ax.set_visible(False)
    return fig, grid


@dataclass(frozen=True)
class PanelSet:
    """Panels whose figure is drawn only once a layout has chosen the size.

    ``render(panels, n_cols, panel_size)`` returns a Figure of exactly
    ``(panel_size[0] * n_cols, panel_size[1] * n_rows)`` inches that shows
    ``panels`` in row-major order; empty slots in the last row are hidden so
    every panel keeps the same width. Producers aggregate once and let
    ``render`` close over the result, so a page draws without recomputing.
    A report layout sizes the panels to its page and can split a long set
    by rows; ``subset`` lays out some panels separately, e.g. one wide panel.
    """

    panels: tuple
    render: Callable[[tuple, int, tuple[float, float]], Figure]
    n_cols: int = 3

    def __post_init__(self):
        object.__setattr__(self, "panels", tuple(self.panels))
        if isinstance(self.n_cols, bool) or not isinstance(self.n_cols, int) or self.n_cols < 1:
            raise ValueError("n_cols must be a positive integer")
        if not self.panels:
            raise ValueError("a PanelSet needs at least one panel")

    @property
    def n_rows(self) -> int:
        return -(-len(self.panels) // self.n_cols)

    def draw(self, panels: Sequence | None = None, panel_size: tuple[float, float] = (5.2, 3.5)) -> Figure:
        """Draw ``panels`` (default: all) at ``panel_size`` inches per panel."""
        chosen = self.panels if panels is None else self._known(panels)
        return self.render(chosen, self.n_cols, tuple(panel_size))

    def subset(self, panels: Sequence, n_cols: int | None = None) -> PanelSet:
        """The same producer limited to ``panels``, optionally on a new grid width."""
        return replace(self, panels=self._known(panels), n_cols=self.n_cols if n_cols is None else n_cols)

    def _known(self, panels: Sequence) -> tuple:
        panels = tuple(panels)
        unknown = [panel for panel in panels if panel not in self.panels]
        if unknown:
            raise ValueError(f"unknown panels: {unknown}")
        if not panels:
            raise ValueError("select at least one panel")
        return panels


def label_outer_panels(axes, bar_axes=None, *, y_titles: bool = False, bar_titles: bool = False,
                       x_titles: bool = False, y_ticks: bool = False, bar_ticks: bool = False) -> None:
    """Keep the chosen axis titles and tick labels on the outer panels only.

    Each flag moves one element: y titles and tick labels to the first
    visible panel of each row, bar-axis titles and tick labels to the last, and
    x titles to the lowest visible panel of each column. Use the tick flags
    only on a shared or fixed scale. Hidden axes are skipped.
    """
    grid = np.asarray(axes, dtype=object)
    grid = grid.reshape(1, -1) if grid.ndim == 1 else grid
    twins = np.full(grid.shape, None, dtype=object) if bar_axes is None else np.asarray(bar_axes, dtype=object).reshape(grid.shape)
    n_rows, n_cols = grid.shape
    for r in range(n_rows):
        visible = [c for c in range(n_cols) if grid[r, c] is not None and grid[r, c].get_visible()]
        for c in visible:
            ax, twin = grid[r, c], twins[r, c]
            if c != visible[0]:
                if y_titles:
                    ax.set_ylabel("")
                if y_ticks:
                    ax.tick_params(axis="y", labelleft=False)
            if x_titles and any(grid[below, c] is not None and grid[below, c].get_visible()
                                for below in range(r + 1, n_rows)):
                ax.set_xlabel("")
            if twin is not None and c != visible[-1]:
                if bar_titles:
                    twin.set_ylabel("")
                if bar_ticks:
                    twin.tick_params(axis="y", labelright=False, length=0)
