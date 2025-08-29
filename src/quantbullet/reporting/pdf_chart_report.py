import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from typing import Optional
from matplotlib.backends.backend_pdf import PdfPages
from .utils import copy_axis, copy_figure

class PdfChartReport:
    def __init__(self, filepath: str, layout=(2,2), figsize: Tuple[int, int]=(11, 8.5)):
        """
        Initializes the PDF report creator.
        Parameters
        ----------
        filepath : str
            Path to save the PDF.
        layout : tuple of int
            (rows, cols) per page.
        figsize : tuple of int, optional
            Size of each PDF page in inches (default is (11, 8.5)).
        """
        self.filepath = filepath
        self.layout = layout  # default / most recent layout
        self.figsize = figsize
        self.pdf = PdfPages(self.filepath)
        self.fig = None # current figure
        self.axes_list = []
        self.current_ax_idx = 0
        self.suptitle = None

    # -------------------------------
    # Internal
    # -------------------------------
    def _add_page( self ):
        """Start a new page with the current layout."""

        # the sqeeze=False ensures we always get a 2D array of axes even we just have 1 row or 1 col
        # the constrained_layout=True helps layout everything but we loss the ability to manually adjust spacing
        self.fig, axes = plt.subplots(*self.layout, figsize=self.figsize, squeeze=False, constrained_layout=False)

        self.axes_list = axes.flatten()
        self.current_ax_idx = 0

        self.fig.subplots_adjust(
            left=0.1, right=0.9, top=0.90, bottom=0.1, hspace=0.4, wspace=0.3
        )
        
        if self.suptitle is not None:
            self.set_suptitle(self.suptitle)

    def _finalize_page(self):
        """Save and close the current page"""
        # don't use bbox_inches='tight' here, it messes up the layout overrides the manual spacing
        if self.fig is not None:

            # hide the unused axes
            if self.current_ax_idx < len(self.axes_list):
                for ax in self.axes_list[self.current_ax_idx:]:
                    ax.set_visible(False)

            self.pdf.savefig( self.fig )
            plt.close( self.fig )
            self.fig = None
            self.axes_list = []
            self.current_ax_idx = 0

    # -------------------------------
    # Public
    # -------------------------------
    def get_next_ax(self) -> plt.Axes:
        """
        Get the next available Axes. Auto-adds a new page if needed.
        Uses the most recent layout.
        """
        if self.fig is None or self.current_ax_idx >= len(self.axes_list):
            self._finalize_page()
            self._add_page()
        ax = self.axes_list[self.current_ax_idx]
        self.current_ax_idx += 1
        return ax

    def new_page(self, layout: Optional[Tuple[int, int]] = None, suptitle: Optional[str] = None):
        """add a new page with a new layout and optionally suptitle."""
        if layout is None:
            layout = self.layout

        if suptitle is not None:
            self.suptitle = suptitle

        self.layout = layout
        self._finalize_page()
        self._add_page()

    def add_figure(self, fig: plt.Figure, suptitle: Optional[str]=None):
        """
        Add a fully constructed figure to the PDF.
        """
        if suptitle:
            fig.suptitle(suptitle)
        self.pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def add_pagebreak( self ):
        """End the current page early."""
        if self.fig is None:
            return
        else:
            self._finalize_page()

    def add_external_axes(self, src_axes, with_legend: bool = True, with_title: bool = True, layout: Optional[Tuple[int, int]] = None, copy_figure_format = False):
        """
        Copy axes (single Axes or array of Axes) into this PDF report grids.
        """

        if layout is not None:
            self.new_page(layout=layout)

        # If it's a single Axes
        if isinstance(src_axes, plt.Axes):
            dst_ax = self.get_next_ax()
            copy_axis(src_axes, dst_ax, with_legend=with_legend, with_title=with_title)

        # If it's an ndarray or list of Axes
        elif isinstance(src_axes, (list, tuple, np.ndarray)):
            flat_axes = np.array(src_axes).flatten()

            if copy_figure_format:
                copy_figure( flat_axes[0].figure, self.fig, include_margins=False, include_spacing=True )

            for ax in flat_axes:
                if isinstance(ax, plt.Axes):
                    dst_ax = self.get_next_ax()
                    copy_axis(ax, dst_ax, with_legend=with_legend, with_title=with_title)

        else:
            raise ValueError("src_axes must be a matplotlib Axes or a collection of Axes.")

    def set_suptitle(self, title: str):
        """Set a suptitle for the current figure"""
        self.fig.suptitle(title, fontsize=16)

    def save(self):
        """
        Finalize and close the PDF.
        """
        self._finalize_page()
        self.pdf.close()