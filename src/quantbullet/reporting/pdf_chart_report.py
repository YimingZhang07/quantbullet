"""Page-composed PDF reports: vector charts, text and tables on report pages."""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.layout_engine import ConstrainedLayoutEngine, TightLayoutEngine
from matplotlib.lines import Line2D
from matplotlib.table import Table
from matplotlib.textpath import text_to_path
from pypdf import PdfReader, PdfWriter

from quantbullet.plot.theme import PRINT_RC

from ._reportlab_utils import PdfColumnGroup, format_pdf_cell
from .formatters import number2string
from .utils import copy_axis

_ALIGNMENTS = ("left", "center", "right")
_CHROME_FIELDS = {"page": 1, "pages": 1, "section": "", "title": ""}
_BULLET_INDENT = 0.18  # inches from the bullet to its text
_CELL_PAD = 0.06  # inches between a cell edge and left- or right-aligned text
_CHROME_RULE_GAP = 0.14  # inches between the body box and a header or footer rule
_CHROME_TEXT_GAP = 0.08  # inches between a rule and its header or footer text


@dataclass(frozen=True)
class ReportStyle:
    """Page geometry, colors and type sizes shared by every page of a report.

    Lengths are inches and type sizes are points. ``margins`` is (left, right,
    top, bottom); the header and footer sit in the top and bottom margins, so
    the body box is the page inside the margins. ``rc`` holds the rcParams for
    pages the report draws (``None`` is ``PRINT_RC``, ``{}`` none), plus
    ``font_family`` when set. ``table_row_scale`` is the table row height as a
    multiple of the table font size.
    """

    page_size: tuple[float, float] = (11.0, 8.5)
    margins: tuple[float, float, float, float] = (0.75, 0.6, 0.65, 0.6)
    font_family: str | None = None
    rc: dict | None = None
    ink: str = "#1c2833"
    muted: str = "#34495e"
    faint: str = "#7f8c8d"
    accent: str = "#1f4e79"
    rule: str = "#d5d8dc"
    zebra: str = "#f4f6f7"
    title_size: float = 16
    subtitle_size: float = 9.5
    heading_size: float = 12
    body_size: float = 9.5
    caption_size: float = 9
    table_size: float = 9
    chrome_size: float = 8
    line_spacing: float = 1.35
    table_row_scale: float = 2.2
    min_chart_height: float = 1.5
    chrome_rules: bool = False

    def __post_init__(self):
        width, height = self.page_size
        left, right, top, bottom = self.margins
        if min(left, right, top, bottom) < 0 or left + right >= width or top + bottom >= height:
            raise ValueError("margins must be non-negative and leave a body box on the page.")


class _Block:
    """Fixed-height content placed by its top edge, in inches above the page bottom."""

    def __init__(self, height: float, space_after: float, place):
        self.height = height
        self.space_after = space_after
        self.place = place


def _chrome_slots(value, name: str) -> tuple[str, str, str]:
    """Normalize a header or footer to (left, center, right) templates."""
    if value is None:
        return ("", "", "")
    slots = (value, "", "") if isinstance(value, str) else tuple(value)
    if len(slots) != 3:
        raise ValueError(f"{name} must be a string or a (left, center, right) tuple.")
    slots = tuple("" if template is None else template for template in slots)
    for template in slots:
        try:
            template.format(**_CHROME_FIELDS)
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(
                f"Bad {name} template {template!r}; its fields are {{page}}, {{pages}}, "
                "{section} and {title}."
            ) from error
    return slots


def _cell_text(value) -> str:
    if isinstance(value, str):
        return value
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, float, np.number)):
        return number2string(value)
    return str(value)


def _table_cells(data, columns, schema, align):
    """Header labels (or None), cell text, and body and header alignments."""
    formats = None
    if schema is not None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("A table schema needs a DataFrame.")
        if any(isinstance(item, PdfColumnGroup) for item in schema):
            raise ValueError("Column groups are not supported in chart report tables.")
        formats = [meta.format for meta in schema]
        header = [meta.display_name or meta.name for meta in schema]
        values = data[[meta.name for meta in schema]].itertuples(index=False, name=None)
        rows = [[format_pdf_cell(value, fmt) for value, fmt in zip(row, formats)] for row in values]
    elif isinstance(data, pd.DataFrame):
        header = [str(column) for column in data.columns]
        rows = [[_cell_text(value) for value in row] for row in data.itertuples(index=False, name=None)]
    else:
        header = None
        rows = [[_cell_text(value) for value in row] for row in data]
    if columns is not None:
        header = [str(column) for column in columns]

    n_cols = len(header) if header is not None else max((len(row) for row in rows), default=0)
    if n_cols == 0:
        raise ValueError("A table needs a header or at least one row.")
    if any(len(row) != n_cols for row in rows):
        raise ValueError(f"Every table row needs {n_cols} cells.")

    aligns = [align] * n_cols if isinstance(align, str) else list(align)
    if len(aligns) != n_cols:
        raise ValueError(f"align needs one alignment or {n_cols}.")
    header_aligns = list(aligns)
    for index, fmt in enumerate(formats or ()):
        if fmt.align:
            aligns[index] = fmt.align.lower()
        if fmt.header_align or fmt.align:
            header_aligns[index] = (fmt.header_align or fmt.align).lower()
    unknown = set(aligns + header_aligns) - set(_ALIGNMENTS)
    if unknown:
        raise ValueError(f"Unknown alignment {sorted(unknown)}; use left, center or right.")
    return header, rows, aligns, header_aligns


class ReportPage:
    """One body page of a :class:`PdfChartReport`, built by stacking blocks.

    Blocks added before :meth:`chart` or :meth:`charts` stack down from the
    title; blocks added after them stack up from the bottom of the body, and
    the charts take the height left between. A page without charts stacks
    every block from the top. Layout runs when the page is written, so chart
    tick labels and titles are measured as drawn; content taller than the body
    raises ``ValueError`` then. Spacing arguments are points.

    Use the page as a context manager: the report's rcParams apply inside the
    ``with`` block, and leaving it writes the page (an exception discards it).
    """

    def __init__(self, report: PdfChartReport, title: str | None = None,
                 subtitle: str | None = None, bookmark_title: str | None = None, *,
                 figure=None, chart_rc: bool = True):
        self._report = report
        self._style = report.style
        self.title = title
        self.subtitle = subtitle
        self.bookmark_title = bookmark_title
        # Charts drawn by the caller outside a ``with`` block keep the caller's
        # rcParams, so titles set later match the ticks set at creation.
        self._chart_rc = report._rc if chart_rc and figure is None else {}
        # Index into _blocks where the chart region sits; an added figure is all chart.
        self._flex = None if figure is None else 0
        if figure is None:
            with plt.rc_context(self._chart_rc):
                figure = plt.figure(figsize=self._style.page_size, layout=ConstrainedLayoutEngine())
        else:
            figure.set_size_inches(*self._style.page_size)
            # Matplotlib keeps a suptitle at the figure top, over the header;
            # the page title takes its place.
            suptitle = figure.get_suptitle()
            if suptitle:
                figure.suptitle("").set_in_layout(False)
                self.title = suptitle if title is None else title
        self.fig = figure
        self._blocks = []
        self._flex_space_after = 0.0
        self._grid_axes = None
        self._chrome = True
        self._in_section = True
        self._closed = False
        self._stack = ExitStack()

    def __enter__(self):
        self._ensure_open()
        self._stack.enter_context(self._rc())
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self._report._drop(self)
            self._discard()
        return False

    @property
    def _body_width(self) -> float:
        left, right, _, _ = self._style.margins
        return self._style.page_size[0] - left - right

    def _rc(self):
        return plt.rc_context(self._report._rc)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("This page has already been written.")

    def heading(self, text: str, *, space_after: float = 4):
        """Add a bold heading inside the page."""
        self._ensure_open()
        self._blocks.append(self._text_block(
            text, size=self._style.heading_size, weight="bold", space_after=space_after))

    def text(self, text: str, *, size: float | None = None, color: str | None = None,
             space_after: float = 10):
        """Add a paragraph wrapped to the body width; newlines start new lines."""
        self._ensure_open()
        self._blocks.append(self._text_block(
            text, size=size or self._style.body_size, color=color, space_after=space_after))

    def bullets(self, items: Sequence[str], *, size: float | None = None,
                space_after: float = 10, item_space: float = 4):
        """Add a bulleted list whose wrapped lines align with the item text."""
        self._ensure_open()
        items = list(items)
        for index, item in enumerate(items):
            self._blocks.append(self._text_block(
                item, size=size or self._style.body_size, marker="•",
                space_after=space_after if index == len(items) - 1 else item_space))

    def caption(self, text: str, *, space_after: float = 8):
        """Add muted small text, e.g. a source line under a chart or table."""
        self._ensure_open()
        self._blocks.append(self._text_block(
            text, size=self._style.caption_size, color=self._style.muted, space_after=space_after))

    def spacer(self, height: float = 12):
        """Add vertical space of ``height`` points."""
        self._ensure_open()
        self._blocks.append(_Block(height / 72, 0.0, lambda top: None))

    def table(self, data, columns: Sequence[str] | None = None, *, schema=None,
              col_widths: Sequence[float] | None = None, align: str | Sequence[str] = "center",
              font_size: float | None = None, zebra: bool = True,
              bold_rows: Sequence[int] | None = None, space_after: float = 10):
        """Add a table that spans the body width.

        ``data`` is a DataFrame or a sequence of rows. A DataFrame's header is
        its columns; rows get a header only from ``columns``, which also
        relabels a DataFrame. ``schema`` (a list of ``PdfColumnMeta``, as in
        ``PdfTextReport.add_df_table``) picks, labels, formats and aligns the
        DataFrame columns; without it numbers go through ``number2string``.
        The index is not shown. ``col_widths`` are relative column widths,
        ``align`` is one alignment or one per column, and ``bold_rows``
        indexes the data rows. Cell text does not wrap.
        """
        self._ensure_open()
        header, rows, aligns, header_aligns = _table_cells(data, columns, schema, align)
        n_cols = len(aligns)
        widths = np.ones(n_cols) if col_widths is None else np.asarray(col_widths, dtype=float)
        if widths.shape != (n_cols,) or not np.all(widths > 0):
            raise ValueError(f"col_widths needs {n_cols} positive widths.")
        widths = widths / widths.sum()
        bold = set()
        for index in bold_rows or ():
            if not -len(rows) <= index < len(rows):
                raise IndexError(f"bold_rows index {index} is outside the {len(rows)} data rows.")
            bold.add(index % len(rows))

        style = self._style
        size = font_size or style.table_size
        row_height = size * style.table_row_scale / 72
        grid = ([(header, header_aligns)] if header is not None else []) + [(row, aligns) for row in rows]
        height = len(grid) * row_height
        page_width, page_height = style.page_size
        left = style.margins[0]

        def place(top):
            with self._rc():
                host = self.fig.add_axes((left / page_width, (top - height) / page_height,
                                          self._body_width / page_width, height / page_height))
                host.set_axis_off()
                table = Table(host, bbox=(0, 0, 1, 1))
                table.auto_set_font_size(False)
                for row_index, (cells, cell_aligns) in enumerate(grid):
                    data_row = row_index - (header is not None)
                    for col_index, text in enumerate(cells):
                        cell = table.add_cell(row_index, col_index, widths[col_index], row_height,
                                              text=text, loc=cell_aligns[col_index], edgecolor=style.rule)
                        cell.set_linewidth(0.4)
                        cell.PAD = _CELL_PAD / (widths[col_index] * self._body_width)
                        label = cell.get_text()
                        label.set(fontsize=size, color=style.ink, parse_math=False)
                        if data_row < 0:
                            cell.set_facecolor(style.accent)
                            label.set(color="white", fontweight="bold")
                        else:
                            cell.set_facecolor(style.zebra if zebra and data_row % 2 else "white")
                            if data_row in bold:
                                label.set_fontweight("bold")
                host.add_table(table)

        self._blocks.append(_Block(height, space_after / 72, place))

    def chart(self, *, space_after: float = 10, **subplot_kw) -> plt.Axes:
        """Add one chart that takes the height left on the page."""
        return self.charts(1, 1, space_after=space_after, subplot_kw=subplot_kw or None)

    def charts(self, nrows: int = 1, ncols: int = 1, *, space_after: float = 10, **subplots_kw):
        """Add a grid of charts that takes the height left on the page.

        Returns what ``Figure.subplots`` returns for the same arguments. A page
        holds one chart region; ``space_after`` separates it from the blocks
        added after it.
        """
        self._ensure_open()
        if self._flex is not None:
            raise ValueError("A page holds one chart region and this page already has one.")
        self._flex = len(self._blocks)
        self._flex_space_after = space_after / 72
        with plt.rc_context(self._chart_rc):
            return self.fig.subplots(nrows, ncols, **subplots_kw)

    def close(self):
        """Lay out and write the page; later calls do nothing."""
        if not self._closed:
            self._report._close_page(self)

    def _text_block(self, text, *, size, weight="normal", color=None, marker=None, space_after):
        style = self._style
        page_width, page_height = style.page_size
        left = style.margins[0]
        indent = _BULLET_INDENT if marker else 0.0
        props = dict(fontsize=size, fontweight=weight, color=color or style.ink, va="top",
                     ha="left", linespacing=style.line_spacing, parse_math=False)
        with self._rc():
            artist = self.fig.text((left + indent) / page_width, 1.0, "", **props)
            artist.set_text(self._wrap(artist, str(text), self._body_width - indent))
            bullet = self.fig.text(left / page_width, 1.0, marker, **props) if marker else None
            height = artist.get_window_extent(self.fig._get_renderer()).height / self.fig.dpi

        def place(top):
            for item in (artist, bullet):
                if item is not None:
                    item.set_y(top / page_height)

        return _Block(height, space_after / 72, place)

    def _rule_block(self, length, *, color, linewidth, space_after):
        page_width, page_height = self._style.page_size
        x0 = self._style.margins[0] / page_width
        height = linewidth / 72

        def place(top):
            y = (top - height / 2) / page_height
            self.fig.add_artist(Line2D([x0, x0 + length / page_width], [y, y],
                                       transform=self.fig.transFigure, color=color,
                                       linewidth=linewidth, solid_capstyle="butt"))

        return _Block(height, space_after / 72, place)

    def _wrap(self, artist, text, width):
        """Break ``text`` into lines no wider than ``width`` inches in the artist's font."""
        # Unhinted widths match the PDF; the Agg renderer's hinted ones run short.
        prop = artist.get_fontproperties()
        limit = width * 72 * 0.99

        def fits(line):
            return text_to_path.get_text_width_height_descent(line, prop, ismath=False)[0] <= limit

        lines = []
        for paragraph in text.split("\n"):
            words = paragraph.split()
            line = words[0] if words else ""
            for word in words[1:]:
                if fits(f"{line} {word}"):
                    line = f"{line} {word}"
                else:
                    lines.append(line)
                    line = word
            lines.append(line)
        return "\n".join(lines)

    def _title_blocks(self):
        style = self._style
        blocks = []
        if self.title:
            blocks.append(self._text_block(self.title, size=style.title_size, weight="bold",
                                           space_after=3 if self.subtitle else 12))
        if self.subtitle:
            blocks.append(self._text_block(self.subtitle, size=style.subtitle_size,
                                           color=style.muted, space_after=12))
        return blocks

    def _layout(self, top: float | None = None):
        """Place the title and blocks in the body box and give the charts the rest."""
        style = self._style
        page_width, page_height = style.page_size
        left, right, top_margin, bottom = style.margins
        chart_axes = list(self.fig.axes)
        titles = self._title_blocks()
        blocks = titles + self._blocks
        split = len(blocks) if self._flex is None else len(titles) + self._flex

        y = page_height - top_margin if top is None else top
        for block in blocks[:split]:
            block.place(y)
            y -= block.height + block.space_after
        name = f"Page {self.title!r}" if self.title else "Page"
        if self._flex is None:
            # Constrained layout warns on a figure without subplots.
            self.fig.set_layout_engine("none")
            content_bottom = y + (blocks[-1].space_after if blocks else 0.0)
            if content_bottom < bottom - 1e-6:
                raise ValueError(f"{name} runs {bottom - content_bottom:.2f} in past the bottom of the body.")
            return

        floor = bottom
        after = blocks[split:]
        for index in range(len(after) - 1, -1, -1):
            if index < len(after) - 1:
                floor += after[index].space_after
            after[index].place(floor + after[index].height)
            floor += after[index].height
        if after:
            floor += self._flex_space_after
        if y - floor < style.min_chart_height:
            raise ValueError(
                f"{name} leaves {max(y - floor, 0.0):.2f} in for its charts; "
                f"they need at least {style.min_chart_height:.2f} in."
            )
        rect = (left / page_width, floor / page_height,
                (page_width - left - right) / page_width, (y - floor) / page_height)
        self._fit_charts(rect, chart_axes)

    def _fit_charts(self, rect, chart_axes):
        engine = self.fig.get_layout_engine()
        if isinstance(engine, ConstrainedLayoutEngine):
            engine.set(rect=rect)
        elif isinstance(engine, TightLayoutEngine):
            engine.set(rect=(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]))
        else:
            # Scale the figure's own layout into the rectangle. Read every
            # position first: twinned Axes move together.
            x0, y0, width, height = rect
            boxes = [(ax, ax.get_position()) for ax in chart_axes]
            for ax, box in boxes:
                ax.set_position((x0 + box.x0 * width, y0 + box.y0 * height,
                                 box.width * width, box.height * height))

    def _write(self, top: float | None = None):
        try:
            with self._rc():
                self._layout(top)
            self._report._write_page(
                self.fig, self.bookmark_title or self.title, rc=self._chart_rc,
                chrome=self._chrome, in_section=self._in_section,
            )
        finally:
            self._discard()

    def _discard(self):
        self._closed = True
        self._stack.close()
        plt.close(self.fig)


class PdfChartReport:
    """Build report pages of vector charts, text and tables, with a PDF outline.

    Each physical page gets one outline entry. Pages after :meth:`add_section`
    become children of that section; without a section they sit at the top
    level. ``header`` and ``footer`` are templates for the top and bottom
    margins: one string (left-aligned) or a (left, center, right) tuple, with
    the fields ``{page}``, ``{pages}``, ``{section}`` and ``{title}``. They are
    stamped on body pages when the report is saved, so ``{pages}`` is the
    final count; title and section divider pages carry none.
    """

    def __init__(
        self,
        filepath: str,
        layout: tuple[int, int] = (2, 2),
        *,
        title: str | None = None,
        style: ReportStyle | None = None,
        header: str | Sequence[str] | None = None,
        footer: str | Sequence[str] | None = None,
    ):
        """Create a report; ``layout`` is the default grid for :meth:`get_next_ax`."""
        self.filepath = filepath
        self._output_path = Path(filepath)
        self.layout = layout
        self.title = title
        self.style = ReportStyle() if style is None else style
        self._header = _chrome_slots(header, "header")
        self._footer = _chrome_slots(footer, "footer")
        rc = PRINT_RC if self.style.rc is None else self.style.rc
        self._rc = {**rc, **({"font.family": self.style.font_family} if self.style.font_family else {})}

        # PdfPages must be closed before pypdf can add the PDF outline. Keep both
        # intermediate PDFs beside the destination so the final replace is atomic.
        fd, render_path = tempfile.mkstemp(
            prefix=".pdf_chart_", suffix=".pdf", dir=self._output_path.parent
        )
        os.close(fd)
        self._render_path = Path(render_path)
        self.pdf = PdfPages(self._render_path)

        self._page = None
        self._grid_next = 0
        self._grid_heading = None
        self._grid_subtitle = None
        self._grid_bookmark = None
        self._grid_page_number = 1
        self._active_section = None
        self._pending_section = None
        self._section_title = ""
        # Each entry is (title, zero-based page index, parent entry index).
        self._bookmarks = []
        # One (has chrome, section title) pair per physical page.
        self._page_chrome = []
        self._closed = False

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("This PDF report has already been saved.")

    def _numbered(self, text):
        if text is None or self._grid_page_number == 1:
            return text
        return f"{text} ({self._grid_page_number})"

    def _add_bookmark(self, title, page_index, parent):
        self._bookmarks.append((title, page_index, parent))
        return len(self._bookmarks) - 1

    def _write_page(self, fig, title, *, rc, chrome=True, in_section=True):
        """Write one physical PDF page and record its outline entry and chrome."""
        page_index = self.pdf.get_pagecount()
        with plt.rc_context(rc or {}):
            self.pdf.savefig(fig)
        parent = None
        if in_section:
            if self._pending_section is not None:
                self._active_section = self._add_bookmark(self._pending_section, page_index, None)
                self._pending_section = None
            parent = self._active_section
        self._page_chrome.append((chrome, self._section_title if in_section else ""))
        return self._add_bookmark(title or f"Page {page_index + 1}", page_index, parent)

    def _finalize_page(self):
        """Write the open page; True when it was a grid page with charts."""
        page, self._page = self._page, None
        if page is None:
            return False
        if page._grid_axes is None:
            page._write()
            return False
        if self._grid_next == 0:
            page._discard()
            return False
        for ax in page._grid_axes[self._grid_next:]:
            ax.set_visible(False)
        page.bookmark_title = self._numbered(self._grid_bookmark or self._grid_heading)
        page._write()
        return True

    def _close_page(self, page):
        if self._page is page:
            self._finalize_page()

    def _drop(self, page):
        if self._page is page:
            self._page = None

    def _check_pending_section(self):
        if self._pending_section is not None:
            raise ValueError(f"Section {self._pending_section!r} has no pages.")

    def page(self, title: str | None = None, subtitle: str | None = None,
             bookmark_title: str | None = None) -> ReportPage:
        """Start a body page; see :class:`ReportPage` for its blocks.

        ``bookmark_title`` overrides ``title`` in the PDF outline; a page with
        neither gets a page-number bookmark.
        """
        self._ensure_open()
        if self._finalize_page():
            self._grid_page_number += 1
        self._page = ReportPage(self, title, subtitle, bookmark_title)
        return self._page

    def get_next_ax(self) -> plt.Axes:
        """Return the next grid Axes, starting a new page when needed."""
        self._ensure_open()
        page = self._page
        if page is None or page._grid_axes is None:
            self._finalize_page()
            self._start_grid()
        elif self._grid_next >= len(page._grid_axes):
            self._finalize_page()
            self._grid_page_number += 1
            self._start_grid()
        ax = self._page._grid_axes[self._grid_next]
        self._grid_next += 1
        return ax

    def _start_grid(self):
        page = ReportPage(self, self._numbered(self._grid_heading), self._grid_subtitle, chart_rc=False)
        page._grid_axes = page.charts(*self.layout, squeeze=False).flatten()
        self._page = page
        self._grid_next = 0

    def new_page(
        self,
        layout: tuple[int, int] | None = None,
        title: str | None = None,
        subtitle: str | None = None,
        bookmark_title: str | None = None,
    ):
        """Configure the next grid without adding a blank page.

        ``bookmark_title`` overrides ``title`` in the PDF outline. A grid
        without either receives a page-number bookmark when written. Overflow
        pages repeat the title with a page counter.
        """
        self._ensure_open()
        self._finalize_page()
        if layout is not None:
            self.layout = layout
        self._grid_heading = title
        self._grid_subtitle = subtitle
        self._grid_bookmark = bookmark_title
        self._grid_page_number = 1

    def set_title(self, title: str):
        """Set the title of the current grid and its future overflow pages."""
        self._ensure_open()
        self._grid_heading = title
        if self._page is not None and self._page._grid_axes is not None:
            self._page.title = self._numbered(title)

    def add_title_page(self, title: str, subtitle: str | None = None,
                       notes: Sequence[str] | None = None):
        """Add a cover page with a top-level outline entry and no header or footer."""
        self._ensure_open()
        if self._finalize_page():
            self._grid_page_number += 1
        style = self.style
        page = ReportPage(self, bookmark_title=title)
        page._chrome = page._in_section = False
        page._blocks.append(page._rule_block(1.2, color=style.accent, linewidth=3, space_after=14))
        page._blocks.append(page._text_block(title, size=28, weight="bold",
                                             space_after=8 if subtitle else 18))
        if subtitle:
            page._blocks.append(page._text_block(subtitle, size=13, color=style.muted, space_after=18))
        for note in notes or ():
            page._blocks.append(page._text_block(note, size=style.body_size, color=style.muted,
                                                 space_after=4))
        page._write(top=style.page_size[1] * 0.64)

    def add_section(self, title: str, subtitle: str | None = None, divider: bool = True):
        """Start a top-level PDF outline section.

        With ``divider`` a centered divider page opens the section. Without it
        the section's outline entry points at its first page, and a section
        that gets no page raises ``ValueError``. ``{section}`` in the header
        and footer is this title until the next section.
        """
        self._ensure_open()
        if not title or not title.strip():
            raise ValueError("Section title must not be empty.")
        self._finalize_page()
        self._check_pending_section()
        self._grid_heading = self._grid_subtitle = self._grid_bookmark = None
        self._grid_page_number = 1
        self._section_title = title
        if not divider:
            self._pending_section = title
            return

        style = self.style
        with plt.rc_context(self._rc):
            fig = plt.figure(figsize=style.page_size)
            try:
                heading = fig.text(0.5, 0.5, title, ha="center", va="center", fontsize=24,
                                   color=style.ink, parse_math=False)
                if subtitle:
                    half = heading.get_window_extent(fig._get_renderer()).height / fig.dpi / 2
                    fig.text(0.5, 0.5 - (half + 0.15) / style.page_size[1], subtitle, ha="center",
                             va="top", fontsize=12, color=style.muted, parse_math=False)
                self._active_section = self._write_page(fig, title, rc=self._rc,
                                                        chrome=False, in_section=False)
            finally:
                plt.close(fig)

    def add_figure(
        self,
        fig: plt.Figure,
        title: str | None = None,
        subtitle: str | None = None,
        bookmark_title: str | None = None,
    ):
        """Resize, write, and close a Figure as one report page.

        The figure's content is fitted into the body below the title: a
        constrained or tight layout gets that rectangle, and any other figure
        has its Axes scaled into it. ``title`` replaces the figure's suptitle,
        which is the title when ``title`` is not given. ``bookmark_title``
        overrides the title in the PDF outline; a page with neither gets a
        page-number bookmark. The figure is drawn under the caller's rcParams.
        """
        self._ensure_open()
        if self._finalize_page():
            self._grid_page_number += 1
        try:
            page = ReportPage(self, title, subtitle, bookmark_title, figure=fig)
        except Exception:
            plt.close(fig)
            raise
        page._write()

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
        layout: tuple[int, int] | None = None,
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
        for src_ax in axes:
            copy_axis(src_ax, self.get_next_ax(), with_legend=with_legend, with_title=with_title)

    def _draw_chrome(self, fig, fields):
        style = self.style
        page_width, page_height = style.page_size
        left, right, top, bottom = style.margins
        xs = (left / page_width, (left + page_width - right) / 2 / page_width,
              (page_width - right) / page_width)
        bands = (
            (self._header, page_height - top + _CHROME_RULE_GAP, _CHROME_TEXT_GAP, "bottom"),
            (self._footer, bottom - _CHROME_RULE_GAP, -_CHROME_TEXT_GAP, "top"),
        )
        for slots, rule_y, offset, va in bands:
            if not any(slots):
                continue
            if style.chrome_rules:
                fig.add_artist(Line2D([xs[0], xs[2]], [rule_y / page_height] * 2,
                                      transform=fig.transFigure, color=style.rule, linewidth=0.6))
            for x, ha, template in zip(xs, _ALIGNMENTS, slots):
                text = template.format(**fields)
                if text:
                    fig.text(x, (rule_y + offset) / page_height, text, ha=ha, va=va,
                             fontsize=style.chrome_size, color=style.faint, parse_math=False)

    def _stamp_chrome(self, writer):
        """Draw headers and footers on their own pages and merge them onto the report."""
        if not any(self._header + self._footer):
            return
        stamped = [index for index, (chrome, _) in enumerate(self._page_chrome) if chrome]
        if not stamped:
            return
        buffer = io.BytesIO()
        with plt.rc_context(self._rc), PdfPages(buffer) as overlay:
            for index in stamped:
                fields = {"page": index + 1, "pages": len(writer.pages),
                          "section": self._page_chrome[index][1], "title": self.title or ""}
                fig = plt.figure(figsize=self.style.page_size)
                try:
                    self._draw_chrome(fig, fields)
                    overlay.savefig(fig, transparent=True)
                finally:
                    plt.close(fig)
        buffer.seek(0)
        for index, overlay_page in zip(stamped, PdfReader(buffer).pages):
            writer.pages[index].merge_page(overlay_page)

    def save(self):
        """Close the Matplotlib PDF, add chrome and outlines, and replace the destination."""
        self._ensure_open()
        output_path = None
        writer = None
        try:
            self._finalize_page()
            self._check_pending_section()
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
            self._stamp_chrome(writer)
            parents = {}
            for entry_index, (title, page_index, parent_index) in enumerate(self._bookmarks):
                parents[entry_index] = writer.add_outline_item(
                    title, page_index,
                    parent=parents[parent_index] if parent_index is not None else None,
                )
            writer.page_mode = "/UseOutlines"
            if self.title:
                writer.add_metadata({"/Title": self.title})
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
