"""Page-oriented PDF reports built from Matplotlib figures."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfWriter

from .utils import copy_axis, copy_figure


class PdfChartReport:
    """Build chart pages with one outline entry per physical PDF page.

    Pages after :meth:`add_section` become children of that section. Without a
    section, chart pages appear at the top level of the PDF outline.
    """

    def __init__(
        self,
        filepath: str,
        layout: Tuple[int, int] = (2, 2),
        figsize: Tuple[int, int] = (11, 8.5),
        corner_text: Optional[str] = None,
    ):
        """Create a chart report with pages measured in inches by ``figsize``."""
        self.filepath = filepath
        self._output_path = Path(filepath)
        self.layout = layout
        self.figsize = figsize
        self.corner_text = corner_text

        # PdfPages must be closed before pypdf can add the PDF outline. Keep both
        # intermediate PDFs beside the destination so the final replace is atomic.
        fd, render_path = tempfile.mkstemp(
            prefix=".pdf_chart_", suffix=".pdf", dir=self._output_path.parent
        )
        os.close(fd)
        self._render_path = Path(render_path)
        self.pdf = PdfPages(self._render_path)

        self.fig = None
        self.axes_list = []
        self.current_ax_idx = 0
        self.suptitle = None
        self._bookmark_title = None
        self._grid_page_number = 1
        self._active_section = None
        # Each entry is (title, zero-based page index, parent entry index).
        self._bookmarks = []
        self._closed = False

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("This PDF report has already been saved.")

    def _grid_title(self):
        if self.suptitle is None:
            return None
        if self._grid_page_number == 1:
            return self.suptitle
        return f"{self.suptitle} ({self._grid_page_number})"

    def _add_page(self):
        """Start a grid page only when an Axes is requested."""
        self.fig, axes = plt.subplots(
            *self.layout, figsize=self.figsize, squeeze=False, constrained_layout=False
        )
        self.axes_list = axes.flatten()
        self.current_ax_idx = 0
        self.fig.subplots_adjust(
            left=0.1, right=0.9, top=0.90, bottom=0.1, hspace=0.4, wspace=0.3
        )
        title = self._grid_title()
        if title is not None:
            self.fig.suptitle(title, fontsize=16)

    def _write_page(self, fig, title: Optional[str], parent: Optional[int]):
        """Write one physical PDF page and record its outline destination."""
        page_index = self.pdf.get_pagecount()
        self.pdf.savefig(fig)
        entry_index = len(self._bookmarks)
        self._bookmarks.append((title or f"Page {page_index + 1}", page_index, parent))
        return entry_index

    def _finalize_page(self):
        """Write the current grid if it has received at least one Axes."""
        if self.fig is None:
            return False

        fig = self.fig
        has_content = self.current_ax_idx > 0
        try:
            if has_content:
                for ax in self.axes_list[self.current_ax_idx:]:
                    ax.set_visible(False)
                if self.corner_text is not None:
                    fig.text(
                        0.99, 0.01, self.corner_text,
                        ha="right", va="bottom", fontsize=10,
                        color="gray", alpha=0.7,
                    )
                title = self._bookmark_title or self.suptitle
                if title and self._grid_page_number > 1:
                    title = f"{title} ({self._grid_page_number})"
                self._write_page(fig, title, self._active_section)
        finally:
            plt.close(fig)
            self.fig = None
            self.axes_list = []
            self.current_ax_idx = 0
        return has_content

    def get_next_ax(self) -> plt.Axes:
        """Return the next grid Axes, starting a new page when needed."""
        self._ensure_open()
        if self.fig is None:
            self._add_page()
        elif self.current_ax_idx >= len(self.axes_list):
            self._finalize_page()
            self._grid_page_number += 1
            self._add_page()
        ax = self.axes_list[self.current_ax_idx]
        self.current_ax_idx += 1
        return ax

    def new_page(
        self,
        layout: Optional[Tuple[int, int]] = None,
        suptitle: Optional[str] = None,
        bookmark_title: Optional[str] = None,
    ):
        """Configure the next grid without adding a blank page.

        ``bookmark_title`` overrides ``suptitle`` in the PDF outline. A grid
        without either title receives a page-number bookmark when written.
        """
        self._ensure_open()
        self._finalize_page()
        if layout is not None:
            self.layout = layout
        self.suptitle = suptitle
        self._bookmark_title = bookmark_title
        self._grid_page_number = 1

    def add_section(self, title: str):
        """Add a centered divider page and start a top-level PDF section."""
        self._ensure_open()
        if not title or not title.strip():
            raise ValueError("Section title must not be empty.")
        self._finalize_page()
        self.suptitle = None
        self._bookmark_title = None
        self._grid_page_number = 1

        fig = plt.figure(figsize=self.figsize)
        try:
            fig.text(0.5, 0.5, title, ha="center", va="center", fontsize=24)
            self._active_section = self._write_page(fig, title, None)
        finally:
            plt.close(fig)

    def add_figure(
        self,
        fig: plt.Figure,
        suptitle: Optional[str] = None,
        bookmark_title: Optional[str] = None,
    ):
        """Resize, write, and close a Figure as one report-sized PDF page.

        ``bookmark_title`` overrides ``suptitle`` in the PDF outline. If both
        are absent, the page receives a page-number bookmark.
        """
        self._ensure_open()
        if self._finalize_page():
            self._grid_page_number += 1
        try:
            fig.set_size_inches(*self.figsize)
            if suptitle:
                fig.suptitle(suptitle)
            self._write_page(fig, bookmark_title or suptitle, self._active_section)
        finally:
            plt.close(fig)

    def add_pagebreak(self):
        """End the current grid early, if it contains a chart."""
        self._ensure_open()
        if self._finalize_page():
            self._grid_page_number += 1

    def add_external_axes(
        self,
        src_axes,
        with_legend: bool = True,
        with_title: bool = True,
        layout: Optional[Tuple[int, int]] = None,
        copy_figure_format: bool = False,
    ):
        """Copy an Axes or collection of Axes into report grid pages."""
        self._ensure_open()
        if isinstance(src_axes, plt.Axes):
            axes = [src_axes]
        elif isinstance(src_axes, (list, tuple, np.ndarray)):
            axes = [ax for ax in np.array(src_axes).flatten() if isinstance(ax, plt.Axes)]
        else:
            raise ValueError("src_axes must be a matplotlib Axes or a collection of Axes.")

        if layout is not None:
            self.new_page(layout=layout)
        for index, src_ax in enumerate(axes):
            dst_ax = self.get_next_ax()
            if index == 0 and copy_figure_format:
                copy_figure(
                    src_ax.figure, self.fig,
                    include_margins=False, include_spacing=True,
                )
            copy_axis(src_ax, dst_ax, with_legend=with_legend, with_title=with_title)

    def set_suptitle(self, title: str):
        """Set the title of the current grid and its future overflow pages."""
        self._ensure_open()
        self.suptitle = title
        if self.fig is not None:
            self.fig.suptitle(self._grid_title(), fontsize=16)

    def save(self):
        """Close the Matplotlib PDF, add outlines, and replace the destination."""
        self._ensure_open()
        output_path = None
        writer = None
        try:
            self._finalize_page()
            page_count = self.pdf.get_pagecount()
            self.pdf.close()
            self._closed = True
            if page_count == 0:
                raise ValueError("Cannot save a PDF report without pages.")

            fd, output_path = tempfile.mkstemp(
                prefix=".pdf_chart_final_", suffix=".pdf", dir=self._output_path.parent
            )
            os.close(fd)
            writer = PdfWriter(clone_from=self._render_path)
            parents = {}
            for entry_index, (title, page_index, parent_index) in enumerate(self._bookmarks):
                parents[entry_index] = writer.add_outline_item(
                    title, page_index,
                    parent=parents[parent_index] if parent_index is not None else None,
                )
            with open(output_path, "wb") as stream:
                writer.write(stream)
            writer.close()
            writer = None
            os.replace(output_path, self._output_path)
        finally:
            if writer is not None:
                writer.close()
            if not self._closed:
                self.pdf.close()
                self._closed = True
            self._render_path.unlink(missing_ok=True)
            if output_path is not None:
                Path(output_path).unlink(missing_ok=True)
