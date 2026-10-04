"""
ReportLab native support fonts are:
- Helvetica
- Helvetica-Bold
- Helvetica-Oblique
- Helvetica-BoldOblique
- Courier
- Courier-Bold
- Courier-Oblique
- Courier-BoldOblique
- Times-Roman
- Times-Bold
- Times-Italic
- Times-BoldItalic
"""
import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    CondPageBreak,
    Flowable,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from quantbullet.plot.panels import PanelSet
from quantbullet.plot.theme import PRINT_RC

from ._reportlab_utils import (
    PdfColumnFormat,
    PdfColumnGroup,
    PdfColumnMeta,
    _flatten_schema,
    apply_heatmap,
    build_table_from_df,
    make_diverging_colormap,
    multi_index_df_to_table_data,
)
from .formatters import number2string
from .utils import copy_axis

class _BookmarkFlowable(Flowable):
    """Zero-size flowable that registers a PDF bookmark and outline entry."""

    width = 0
    height = 0

    def __init__(self, key: str, title: str, level: int = 0, closed: bool = False):
        super().__init__()
        self.key = key
        self.title = title
        self.level = level
        self.closed = closed

    def draw(self):
        self.canv.bookmarkPage(self.key)
        self.canv.addOutlineEntry(self.title, self.key, level=self.level, closed=self.closed)


_FRAME_PADDING = 6  # SimpleDocTemplate's Frame pads each side by 6pt


def _place_figure(canv, fig, width, height, dpi):
    """Rasterize ``fig`` and draw it 1:1 in a ``width`` x ``height`` (pt) box."""
    fig_w, fig_h = fig.get_size_inches()
    if abs(fig_w * 72 - width) > 0.5 or abs(fig_h * 72 - height) > 0.5:
        plt.close(fig)
        raise ValueError(
            f"draw returned a {fig_w:.2f}x{fig_h:.2f} in figure for a {width / 72:.2f}x{height / 72:.2f} in "
            "slot; draw at the size given so text keeps its point size")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)  # the whole figure box: no bbox_inches="tight"
    plt.close(fig)
    buf.seek(0)
    canv.drawImage(ImageReader(PILImage.open(buf).convert("RGB")), 0, 0, width, height)


class _FigureFlowable(Flowable):
    """A figure drawn at the size its frame offers and placed 1:1."""

    def __init__(self, draw, *, height, min_height, dpi, rc):
        super().__init__()
        self._draw_figure, self._fixed_height, self._min_height = draw, height, min_height
        self._dpi, self._rc = dpi, rc

    def wrap(self, availWidth, availHeight):
        height = self._fixed_height if self._fixed_height is not None else max(availHeight, self._min_height)
        self.width, self.height = availWidth, height
        return self.width, self.height

    def split(self, availWidth, availHeight):
        return []  # a figure moves to the next frame whole

    def draw(self):
        with plt.rc_context(self._rc):
            fig = self._draw_figure(self.width / 72, self.height / 72)
            _place_figure(self.canv, fig, self.width, self.height, self._dpi)


class _PanelGridFlowable(Flowable):
    """A PanelSet whose rows fill the frame; it splits between rows across frames."""

    def __init__(self, panels: PanelSet, *, min_panel_height, max_panel_aspect, dpi, rc,
                 panel_height=None, continued=None):
        super().__init__()
        self.panels = panels
        self._min_panel_height, self._max_aspect = min_panel_height, max_panel_aspect
        self._dpi, self._rc = dpi, rc
        self._panel_height, self._continued = panel_height, continued

    def _panel_size(self, availWidth, availHeight, n_rows):
        width = availWidth / self.panels.n_cols
        cap = width * self._max_aspect if self._panel_height is None else self._panel_height
        return width, max(self._min_panel_height, min(cap, availHeight / n_rows))

    def wrap(self, availWidth, availHeight):
        self._size = self._panel_size(availWidth, availHeight, self.panels.n_rows)
        self.width, self.height = availWidth, self._size[1] * self.panels.n_rows
        return self.width, self.height

    def split(self, availWidth, availHeight):
        fit = int((availHeight + 1e-6) // self._min_panel_height)
        if fit < 1 or fit >= self.panels.n_rows:
            return []
        cut = fit * self.panels.n_cols
        _, panel_height = self._panel_size(availWidth, availHeight, fit)
        options = dict(min_panel_height=self._min_panel_height, max_panel_aspect=self._max_aspect,
                       dpi=self._dpi, rc=self._rc, continued=self._continued)
        head = _PanelGridFlowable(self.panels.subset(self.panels.panels[:cut]), **options)
        # The continuation keeps this page's panel height when it fits.
        tail = _PanelGridFlowable(self.panels.subset(self.panels.panels[cut:]), panel_height=panel_height, **options)
        pieces = [head, PageBreak()]
        if self._continued:
            style = ParagraphStyle(name="ContinuedStyle", fontName="Helvetica-Oblique", fontSize=9,
                                   textColor=colors.grey, spaceAfter=4)
            pieces.append(Paragraph(PdfTextReport._escape_html(f"{self._continued} (continued)"), style))
        return pieces + [tail]

    def draw(self):
        with plt.rc_context(self._rc):
            fig = self.panels.draw(panel_size=(self._size[0] / 72, self._size[1] / 72))
            _place_figure(self.canv, fig, self.width, self.height, self._dpi)


def primary_axes(fig):
    """Visible subplot axes, excluding twinx count axes."""
    primaries = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        siblings = list(ax._twinned_axes.get_siblings(ax))
        if siblings and ax is not siblings[0]:
            continue
        primaries.append(ax)
    return primaries


def copy_panel(src_ax, dst_ax):
    """Copy one panel, keeping a twinx count axis overlaid on the curves."""
    copy_axis(src_ax, dst_ax)
    twins = [ax for ax in src_ax._twinned_axes.get_siblings(src_ax) if ax is not src_ax and ax.get_visible()]
    if not twins:
        return
    dst_twin = dst_ax.twinx()
    copy_axis(twins[0], dst_twin, with_legend=False)
    dst_ax.set_zorder(dst_twin.get_zorder() + 1)
    dst_ax.patch.set_visible(False)
    dst_twin.grid(False)


class PdfTextReport:
    def __init__( self, 
                  file_path:str, 
                  page_size:tuple=None, 
                  report_title:str=None, 
                  margins:tuple=(36,36,36,36), 
                  page_numbering:bool=True ):

        if page_size is None:
            self.page_size = landscape(letter)
        else:
            self.page_size = (page_size[0] * inch, page_size[1] * inch)

        # sometimes file_path is a Path object
        if not isinstance(file_path, str):
            file_path = str(file_path)

        self.doc = SimpleDocTemplate(
            file_path,
            pagesize=self.page_size,
            leftMargin=margins[0],
            rightMargin=margins[1],
            topMargin=margins[2],
            bottomMargin=margins[3]
        )

        self.story = []
        self.report_title = report_title
        if report_title is not None:
            self.add_centered_text( report_title, font_size=14, space_after=12 )

        self.page_numbering = page_numbering

    def add_page_break(self):
        self.story.append( PageBreak() )
        
    @staticmethod
    def _normalize_table_data( data:list ):
        """Ensure all values if its a number, do number2string.
        
        Parameters
        ----------
        data : list
            2D list of table data.
        """
        
        normalized = []
        for row in data:
            new_row = []
            for v in row:
                if np.issubdtype(type(v), np.number):
                    new_row.append( number2string(v) )
                else:
                    new_row.append( str(v) )
            normalized.append(new_row)
        return normalized
        
    def add_two_col_table( self, data:list, col_widths:list=None, style:list=None, header:bool=True ):
        data = self._normalize_table_data(data)
        if col_widths is None:
            col_widths = [200, 200]
        t = Table(data, colWidths=col_widths)
        if style is None:
            style = [
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,-1), 'Courier')
            ]
            if header:
                style.append( ('BACKGROUND', (0,0), (-1,0), colors.lightgrey) )
                style.append( ('FONTNAME', (0,0), (-1,0), 'Courier-Bold') )
        t.setStyle( TableStyle(style) )
        self.story.append(t)
        self.story.append( Spacer(1, 12) )

    def _bold_rows_cols_styles( self, nrows:int, ncols:int, bold_rows:list[int]=None, bold_cols:list[int]=None ):
        styles = []
        if bold_rows:
            rows = [(r if r >= 0 else nrows + r) for r in bold_rows]
            for r in rows:
                styles.append(("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"))
        if bold_cols:
            cols = [(c if c >= 0 else ncols + c) for c in bold_cols]
            for c in cols:
                styles.append(("FONTNAME", (c, 0), (c, -1), "Helvetica-Bold"))
        return styles

    def add_df_table( self, 
                      df, 
                      schema:list[PdfColumnMeta], 
                      space_after:int=12, 
                      font_size:int=8, 
                      bold_rows=None, 
                      bold_cols=None, 
                      heatmap_all:bool=False, 
                      color_map=None, 
                      cmap_vmid=None,
                      col_widths=None ):
        """Add a DataFrame as a table to the PDF.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to render as a table.
        schema : list of PdfColumnMeta
            Metadata for each column, including formatting and colormap info.
        space_after : int, optional
            Space after the table in points, by default 12.
        font_size : int, optional
            Font size for the table text, by default 8.
        bold_rows : list of int, optional
            List of row indices to bold, by default None.
        bold_cols : list of int, optional
            List of column indices to bold, by default None.
        heatmap_all : bool, optional
            Whether to apply a heatmap to all numeric columns, by default False.
        color_map : callable, optional
            A colormap function that takes a float in [0, 1] and returns a color string, by default None.
        cmap_vmid : float, optional
            The midpoint value for the colormap, by default None.
        """
        tbl = build_table_from_df( df, schema, col_widths=col_widths )
        flat_cols, group_spans = _flatten_schema(schema)
        header_offset = 2 if group_spans else 1
        nrows, ncols = len(df) + header_offset, len(flat_cols)

        # deal with the heatmap if needed
        if heatmap_all:
            if color_map is None:
                color_map = make_diverging_colormap( high_color="#639567", mid_color="#aed6b2", low_color=(1, 1, 1) )
            _styles = apply_heatmap( table_data=tbl._cellvalues, 
                                    row_range=(1, nrows-1), 
                                    col_range=(0, ncols-1),
                                    cmap=color_map,
                                    vmid=cmap_vmid )
            tbl.setStyle( TableStyle(_styles) )

        # tbl has the style already applied lets append the font size style
        styles = [ ( "FONTSIZE", ( 0, 0 ), ( -1, -1 ), font_size) ]
        styles += self._bold_rows_cols_styles( nrows, ncols, bold_rows, bold_cols )

        tbl.setStyle(TableStyle(styles))
        self.story.append( tbl )
        self.story.append( Spacer( 1, space_after ) )

    def add_df_table_breakdown( self, df, schema, nrows=20, space_between=8, space_after=12, font_size=8 ):
        # This is to add a large table, and we want to repeat the table for every several rows
        # the space between the smaller tables is controlled by the space_between parameter
        # the last table will have space_after applied
        total_rows = len( df )
        first_table = True
        table_widths = None

        # even if the table is broken down, we still need to compute the colormap vmin/vmax
        flat_cols, _ = _flatten_schema(schema)
        for col_schema in flat_cols:
            if col_schema.format.colormap is not None:
                # we need to compute vmin and vmax for the colormap
                col_values = df[ col_schema.name ].dropna().values
                vmin = np.min( col_values )
                vmax = np.max( col_values )
                if col_schema.format.vmin is None:
                    col_schema.format.vmin = vmin
                if col_schema.format.vmax is None:
                    col_schema.format.vmax = vmax

        for start_row in range( 0, total_rows, nrows ):
            end_row = min( start_row + nrows, total_rows )
            sub_df = df.iloc[ start_row:end_row ].copy()
            self.add_df_table( sub_df, schema, space_after=( space_after if end_row == total_rows else space_between ), font_size=font_size, col_widths=table_widths )

            # for the first table, we need to wrap it to fit the page size
            # then we can forward its size to the later tables for better performance
            if first_table:
                first_table = False
                t = self.story[-2]  # the last added table
                t.wrap(self.page_size[0], self.page_size[1])
                table_widths = t._colWidths

    def compute_col_widths_for_df( self, df, schema, font_size=8 ):
        if df is None or df.empty:
            return None

        # Build a temporary table from the full df
        tbl = build_table_from_df(df, schema)

        # Use the same base font size; this affects width calculation
        tbl.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ]))

        # Let ReportLab compute column widths.
        # If you have a "frame width" instead of full page width, use that here.
        tbl.wrap(self.doc.width, self.doc.height)

        # _colWidths is what ReportLab decided was best
        return list(tbl._colWidths)

    def compute_table_left_indent( self, df, schema, font_size: int = 8, col_widths: list[float] | None = None ):
        """Compute left indent to align text with a centered table."""
        if col_widths is None:
            col_widths = self.compute_col_widths_for_df( df, schema, font_size=font_size )
        if not col_widths:
            return 0, col_widths
        table_width = sum( col_widths )
        left_indent = max( ( self.doc.width - table_width ) / 2, 0 )
        return left_indent, col_widths

    def add_multiindex_df_table( self, df, font_size:int=6, space_after:int=12, 
                                 heatmap_cols:bool=False, heatmap_all:bool=False, 
                                 bold_rows=None, bold_cols=None, heatmap_cmap:callable=None, heatmap_selected_cols:list=None):
        """Add a MultiIndex DataFrame as a table to the PDF.
        
        This is a less configurable function than the regular add_df_table, due to the complexity of MultiIndex tables.
        This table has focused on displaying the MultiIndex structure clearly. So you need to have
            1) the dataframe formatted as strings in the way they are to be displayed
            2) the None values filled in as empty strings "" so that the table looks clean
        """
        table_data, spans = multi_index_df_to_table_data( df )
        nrow_levels = df.index.nlevels
        ncol_levels = df.columns.nlevels

        heatmap_styles = []
        if heatmap_cols:
            # heatmap the whole table but column by column
            n_rows = len(table_data)
            n_cols = len(table_data[0])
            for col in range(nrow_levels, n_cols):
                _styles = apply_heatmap( table_data=table_data, 
                                         row_range=(nrow_levels, n_rows-1), 
                                         col_range=(col, col),
                                         cmap=make_diverging_colormap(),
                                         vmid=0 )
                heatmap_styles.extend(_styles)

        if heatmap_selected_cols:
            n_rows = len(table_data)
            n_cols = len(table_data[0])
            for col in heatmap_selected_cols:
                if col < nrow_levels or col >= n_cols:
                    continue
                _styles = apply_heatmap(
                    table_data=table_data,
                    row_range=(nrow_levels, n_rows - 1),
                    col_range=(col, col),
                    cmap=heatmap_cmap
                    if heatmap_cmap
                    else make_diverging_colormap(high_color="#63be7b", mid_color=(1, 1, 1), low_color=(1, 1, 1)),
                    vmid=None,
                )
                heatmap_styles.extend(_styles)

        if heatmap_all:
            # heatmap the whole table
            n_rows = len(table_data)
            n_cols = len(table_data[0])
            _styles = apply_heatmap( table_data=table_data, 
                                            row_range=(nrow_levels, n_rows-1),
                                            col_range=(nrow_levels, n_cols-1),
                                            cmap=make_diverging_colormap( high_color="#63be7b", mid_color=(1,1,1), low_color="#f8696b" ),
                                            vmid=10 )
            heatmap_styles.extend(_styles)

        main_styles = [
            ("GRID",       (0, 0),                   (-1, -1),                0.5,      "black"),
            ("ALIGN",      (0, 0),                   (-1, -1),                "CENTER"),
            ("VALIGN",     (0, 0),                   (0, -1),                 "MIDDLE"),
            ("ALIGN",      (nrow_levels, ncol_levels), (-1, -1),              "RIGHT"),
            ("BACKGROUND", (0, 0),                   (-1, ncol_levels - 1),   "#d9d9d9"),
            ("FONTSIZE",   (0, 0),                   (-1, -1),                font_size),
        ]
        extended_styles = main_styles + spans + heatmap_styles

        nrows = len(table_data)
        ncols = len(table_data[0])
        extended_styles += self._bold_rows_cols_styles( nrows, ncols, bold_rows, bold_cols )

        table_style = TableStyle( extended_styles )

        tbl = Table(table_data, style=table_style)

        self.story.append( tbl )
        self.story.append( Spacer( 1, space_after ) )

    def add_text( self, text: str, font_size: int = 10, space_after: int = 12, alignment: int = 0, left_indent: int = 0 ):
        """Add a left-aligned text paragraph."""
        style = ParagraphStyle(
            name        = "NormalStyle",
            fontName    = "Helvetica",
            fontSize    = font_size,
            alignment   = alignment,
            leftIndent  = left_indent,
        )
        p = Paragraph( text, style=style )
        self.story.append( p )
        self.story.append( Spacer( 1, space_after ) )

    def add_pre(self, text: str, font_size: int = 8, space_after: int = 12, left_indent: int = 0):
        style = ParagraphStyle(
            name="CodeBlock",
            fontName="Courier",
            fontSize=font_size,
            leftIndent=left_indent,
            leading=font_size * 1.2,
        )
        self.story.append(Preformatted(text, style))
        self.story.append(Spacer(1, space_after))

    def add_table_footnote(self, text:str, font_size:int=8, space_after:int=0, alignment:int=0):
        """Add a footnote text paragraph, typically after a table."""
        style = ParagraphStyle(
            name="FootnoteStyle",
            fontName="Helvetica-Oblique",
            fontSize=font_size,
            textColor=colors.grey,
            leftIndent=25,
            alignment=alignment  # 0=left, 1=center, 2=right
        )
        p = Paragraph(text, style=style)
        self.story.append(p)
        self.story.append(Spacer(1, space_after))
        
    def add_centered_text(self, text:str, font_size:int=12, space_after:int=12):
        """Add a centered text paragraph."""
        style = ParagraphStyle(
            name="CenteredStyle",
            fontName="Helvetica",
            fontSize=font_size,
            alignment=TA_CENTER
        )
        p = Paragraph(text, style=style)
        self.story.append(p)
        self.story.append(Spacer(1, space_after))

    # ---------------------------------------------------------------------------
    # Convenience text helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape ``&``, ``<``, ``>`` so plain text is safe inside ReportLab XML."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    _HEADING_SIZES = {1: 14, 2: 12, 3: 10}

    def add_heading(
        self,
        text: str,
        level: int = 1,
        bookmark: bool = True,
        space_after: int = 6,
    ):
        """Add a bold section heading with optional PDF bookmark.

        Parameters
        ----------
        text : str
            Plain-text heading content (``&``, ``<``, ``>`` are auto-escaped).
        level : int, optional
            Heading level: 1 (largest, 14pt), 2 (12pt), or 3 (10pt).
        bookmark : bool, optional
            If True, a PDF outline bookmark is registered at this position.
        space_after : int, optional
            Vertical space after the heading in points.
        """
        font_size = self._HEADING_SIZES.get(level, 10)
        safe = self._escape_html(text)
        start = len(self.story)

        if bookmark:
            bookmark_level = max(level - 1, 0)
            self.add_bookmark(text, level=bookmark_level)

        style = ParagraphStyle(
            name="HeadingStyle",
            fontName="Helvetica-Bold",
            fontSize=font_size,
        )
        paragraph = Paragraph(safe, style)
        paragraph._heading_text = text
        self.story.append(paragraph)
        self.story.append(Spacer(1, space_after))
        # add_figure and add_figure_grid keep these on the page with the figure.
        for flowable in self.story[start:]:
            flowable._heading_part = True

    def add_body(
        self,
        text: str,
        font_size: int = 9,
        space_after: int = 6,
        alignment: int = 0,
        left_indent: int = 0,
    ):
        """Add a body-text paragraph with sensible report defaults.

        This is a thin wrapper around :meth:`add_text` with smaller font size
        and tighter spacing suited for report body content.  The *text* may
        contain ReportLab inline markup (``<b>``, ``<i>``, ``<br/>``, etc.).

        Parameters
        ----------
        text : str
            Paragraph content (may include ReportLab XML markup).
        font_size : int, optional
            Font size in points (default 9).
        space_after : int, optional
            Vertical space after the paragraph in points (default 6).
        alignment : int, optional
            0 = left, 1 = centre, 2 = right (default 0).
        left_indent : int, optional
            Left indent in points (default 0).
        """
        self.add_text(
            text,
            font_size=font_size,
            space_after=space_after,
            alignment=alignment,
            left_indent=left_indent,
        )

    def add_kv_line(
        self,
        label_or_pairs: "str | dict[str, str]",
        value: str | None = None,
        *,
        separator: str = " &nbsp;|&nbsp; ",
        font_size: int = 9,
        space_after: int = 4,
    ):
        """Add one or more **bold label : value** pairs on a single line.

        Two calling conventions are supported::

            report.add_kv_line("MAE", "31.2 bps")
            report.add_kv_line({"MAE": "31.2 bps", "RMSE": "45.1 bps"})

        Parameters
        ----------
        label_or_pairs : str or dict
            A single label string (requires *value*) **or** a ``{label: value}``
            dict for multiple pairs rendered on one line.
        value : str, optional
            The value string when *label_or_pairs* is a single label.
        separator : str, optional
            HTML separator inserted between multiple pairs.
        font_size : int, optional
            Font size in points (default 9).
        space_after : int, optional
            Vertical space after the line in points (default 4).
        """
        if isinstance(label_or_pairs, dict):
            pairs = label_or_pairs
        else:
            if value is None:
                raise ValueError("value is required when label_or_pairs is a string")
            pairs = {label_or_pairs: value}

        fragments = [f"<b>{k}:</b> {v}" for k, v in pairs.items()]
        html = separator.join(fragments)
        self.add_text(html, font_size=font_size, space_after=space_after)

    def add_list(
        self,
        items: list[str],
        ordered: bool = False,
        font_size: int = 9,
        space_after: int = 6,
        left_indent: int = 18,
        bullet_font_size: int | None = None,
    ):
        """Add a bulleted or numbered list.

        Parameters
        ----------
        items : list of str
            List items.  Each may contain ReportLab inline markup.
        ordered : bool, optional
            ``True`` for a numbered list, ``False`` for bullets (default).
        font_size : int, optional
            Font size for item text (default 9).
        space_after : int, optional
            Vertical space after the whole list in points (default 6).
        left_indent : int, optional
            Left indent for the list in points (default 18).
        bullet_font_size : int, optional
            Font size for the bullet / number.  Defaults to *font_size*.
        """
        if bullet_font_size is None:
            bullet_font_size = font_size

        style = ParagraphStyle(
            name="ListItemStyle",
            fontName="Helvetica",
            fontSize=font_size,
            leading=font_size * 1.4,
        )

        list_items = [ListItem(Paragraph(item, style)) for item in items]

        lf = ListFlowable(
            list_items,
            bulletType="1" if ordered else "bullet",
            bulletFontSize=bullet_font_size,
            leftIndent=left_indent,
            bulletOffsetY=-1,
            start=1 if ordered else None,
        )
        self.story.append(lf)
        self.story.append(Spacer(1, space_after))

    def content_size_inches(self) -> tuple[float, float]:
        """Width and height in inches that flowables get on an empty page (inside the frame padding)."""
        width, height = self._frame_size()
        return width / 72.0, height / 72.0

    def _frame_size(self) -> tuple[float, float]:
        width, height = self.get_page_dimensions()
        return width - 2 * _FRAME_PADDING, height - 2 * _FRAME_PADDING

    def add_figure(self, draw, *, height: float | None = None, min_height: float = 2.0,
                   dpi: int = 200, rc: dict | None = None, space_after: float = 6):
        """Add a figure drawn at the size of its slot and placed 1:1.

        ``draw(width, height)`` gets the slot size in inches and must return a
        Figure of exactly that size, e.g. via ``figsize=(width, height)``; text
        then keeps its point size on the page. The slot spans the frame width.
        ``height`` (inches) fixes its height; ``None`` takes the rest of the
        page, at least ``min_height`` inches. ``draw`` runs when the PDF is
        built, under ``rc`` (default ``PRINT_RC``; ``{}`` for none), so give
        themed plots the matching ``PRINT_THEME``. Headings added just before
        stay on the page with the figure. ``space_after`` (points) follows a
        fixed-height figure.
        """
        _, frame_height = self._frame_size()
        if height is not None and not 0 < height * 72 <= frame_height:
            raise ValueError(f"height must be in (0, {frame_height / 72:.2f}] inches")
        if not 0 < min_height * 72 <= frame_height:
            raise ValueError(f"min_height must be in (0, {frame_height / 72:.2f}] inches")
        start, heading_height, _ = self._heading_run()
        below = (height if height is not None else min_height) * 72
        self.story.insert(start, CondPageBreak(min(heading_height + below, frame_height)))
        self.story.append(_FigureFlowable(draw, height=None if height is None else height * 72,
                                          min_height=min_height * 72, dpi=dpi, rc=PRINT_RC if rc is None else rc))
        if height is not None and space_after:
            self.story.append(Spacer(1, space_after))

    def add_figure_grid(self, panels: PanelSet, *, min_panel_height: float = 1.75,
                        max_panel_aspect: float = 0.75, dpi: int = 200, rc: dict | None = None):
        """Add a ``PanelSet`` sized to the page.

        Panels are ``frame width / panels.n_cols`` wide; their rows share the
        remaining page height, each between ``min_panel_height`` inches and
        ``max_panel_aspect`` times the panel width. A grid that fits on a fresh
        page moves there with its headings; a taller one fills this page and
        continues by whole rows at the same panel size, under the last heading
        marked "(continued)". Each page is drawn from the ``PanelSet``, so
        styles, count axes and shared scales stay intact. Drawing happens when
        the PDF is built, under ``rc`` as in ``add_figure``.
        """
        _, frame_height = self._frame_size()
        min_panel = min_panel_height * 72
        if not 0 < min_panel <= frame_height:
            raise ValueError(f"min_panel_height must be in (0, {frame_height / 72:.2f}] inches")
        if not max_panel_aspect > 0:
            raise ValueError("max_panel_aspect must be positive")
        start, heading_height, heading = self._heading_run()
        whole = panels.n_rows * min_panel
        below = whole if heading_height + whole <= frame_height else min_panel
        self.story.insert(start, CondPageBreak(min(heading_height + below, frame_height)))
        self.story.append(_PanelGridFlowable(
            panels, min_panel_height=min_panel, max_panel_aspect=max_panel_aspect, dpi=dpi,
            rc=PRINT_RC if rc is None else rc, continued=heading,
        ))

    def _heading_run(self) -> tuple[int, float, str | None]:
        """Start index, height (pt) and last text of the headings that end the story."""
        start = len(self.story)
        while start > 0 and getattr(self.story[start - 1], "_heading_part", False):
            start -= 1
        width, height = self._frame_size()
        run = self.story[start:]
        texts = [f._heading_text for f in run if hasattr(f, "_heading_text")]
        return start, sum(f.wrap(width, height)[1] for f in run), texts[-1] if texts else None

    def add_matplotlib_figure(self, fig, width_fraction=1, space_after=12, dpi=600, reserve_height=0):
        """Add a matplotlib figure at its printed size.

        The PNG is placed at ``pixels / dpi`` inches, so a figure drawn to the
        content width keeps its matplotlib point sizes. It is reduced only when
        it is wider or taller than the page. Prefer ``add_figure``, which draws
        at the size of the slot.
        """
        available_w, available_h = self._frame_size()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        pil_img = PILImage.open(buf)
        px_w, px_h = pil_img.size
        buf.seek(0)

        draw_w = px_w / dpi * 72
        draw_h = px_h / dpi * 72
        max_w = available_w * width_fraction
        max_h = available_h - space_after - reserve_height - 5
        if draw_w > max_w or draw_h > max_h:
            scale = min(max_w / draw_w, max_h / draw_h)
            draw_w *= scale
            draw_h *= scale

        self.story.append(Image(buf, width=draw_w, height=draw_h))
        plt.close(fig)
        self.story.append(Spacer(1, space_after))
        
    def add_matplotlib_figure_paged(
        self,
        fig,
        n_cols: int,
        width_fraction: float = 0.95,
        space_after: float = 12,
        dpi: int = 150,
        title: str | None = None,
    ):
        """Add a multi-panel figure, automatically splitting across pages if needed.

        Split pages are rebuilt by copying axes, which drops theme styling.
        Prefer ``add_figure_grid`` with a ``PanelSet``, which redraws each page.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Source figure with a grid of subplots.
        n_cols : int
            Number of subplot columns in the grid.
        title : str, optional
            Suptitle added to every page (with page numbering when split).
        """
        import math

        visible_axes = primary_axes(fig)
        if not visible_axes:
            plt.close(fig)
            return

        n_axes = len(visible_axes)
        n_rows_total = math.ceil(n_axes / n_cols)

        fig_w, fig_h = fig.get_size_inches()
        per_row_h = fig_h / max(n_rows_total, 1)
        per_col_w = fig_w / max(n_cols, 1)

        _, available_h = self._frame_size()
        # Figure inches are the printed size. Compare that height with the page.
        available_h_inches = (available_h - space_after - 5) / 72.0
        max_rows_per_page = max(1, int(available_h_inches / per_row_h))

        if n_rows_total <= max_rows_per_page:
            if title:
                fig.suptitle(title, fontsize=13, y=1.02)
            self.add_matplotlib_figure(fig, width_fraction, space_after, dpi)
            return

        rows = []
        for i in range(0, n_axes, n_cols):
            rows.append(visible_axes[i : i + n_cols])

        for page_idx in range(0, len(rows), max_rows_per_page):
            chunk = rows[page_idx : page_idx + max_rows_per_page]
            n_rows_chunk = len(chunk)
            new_fig, new_axes = plt.subplots(
                n_rows_chunk, n_cols,
                figsize=(per_col_w * n_cols, per_row_h * n_rows_chunk),
                squeeze=False,
            )
            for r, row_axes in enumerate(chunk):
                for c, src_ax in enumerate(row_axes):
                    copy_panel(src_ax, new_axes[r][c])
                for c in range(len(row_axes), n_cols):
                    new_axes[r][c].set_visible(False)

            if title:
                page_num = page_idx // max_rows_per_page + 1
                total_pages = math.ceil(len(rows) / max_rows_per_page)
                suffix = f" ({page_num}/{total_pages})" if total_pages > 1 else ""
                new_fig.suptitle(title + suffix, fontsize=13, y=1.02)

            new_fig.tight_layout()
            self.add_matplotlib_figure(new_fig, width_fraction, space_after, dpi)
            if page_idx + max_rows_per_page < len(rows):
                self.add_page_break()

        plt.close(fig)

    # -----------------
    # Page dimensions
    # -----------------
    def get_page_dimensions(self):
        """Return usable (width, height) after margins in points."""
        page_w, page_h = self.doc.pagesize
        usable_w = page_w - self.doc.leftMargin - self.doc.rightMargin
        usable_h = page_h - self.doc.topMargin - self.doc.bottomMargin
        return usable_w, usable_h
        
    def save( self ):
        if self.page_numbering:
            self.doc.build( self.story, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number )
        else:
            self.doc.build( self.story )
            
    def _add_page_number(self, canvas, doc):
        """Add page number at bottom center."""
        page_num = canvas.getPageNumber()
        text = f"{ page_num }"
        width, height = self.doc.pagesize
        canvas.setFont( "Helvetica", 9 )
        canvas.drawCentredString( width / 2.0, 15, text )  # y=15 points from bottom

    _bookmark_counter = 0

    def add_bookmark(self, title: str, level: int = 0, closed: bool = False, key: str | None = None):
        """Add a PDF outline bookmark at the current position.

        Parameters
        ----------
        title : str
            Caption shown in the PDF outline/bookmarks panel.
        level : int, optional
            Nesting depth (0 = top-level, 1 = sub-section, etc.). It is an error
            to jump down more than one level at a time (e.g. 0 then 2).
        closed : bool, optional
            Whether the node starts collapsed in the outline tree.
        key : str, optional
            Unique internal key. Auto-generated if omitted.
        """
        if key is None:
            PdfTextReport._bookmark_counter += 1
            key = f"_bm_{PdfTextReport._bookmark_counter}"
        self.story.append(_BookmarkFlowable(key, title, level=level, closed=closed))

    def add_spacer( self, height: int = 12 ):
        """Add a vertical spacer."""
        self.story.append( Spacer( 1, height ) )