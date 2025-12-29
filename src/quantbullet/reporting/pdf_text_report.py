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
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Preformatted,
)

from .formatters import number2string
from ._reportlab_utils import PdfColumnFormat, PdfColumnMeta, build_table_from_df, multi_index_df_to_table_data, apply_heatmap, make_diverging_colormap
from ..plot.colors import ColorEnum

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
        nrows, ncols = len(df) + 1, len(schema)

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
        for col_schema in schema:
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
        available_width = self.page_size[0]
        available_height = self.page_size[1]
        tbl.wrap(available_width, available_height)

        # _colWidths is what ReportLab decided was best
        return list(tbl._colWidths)

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

    def add_matplotlib_figure(self, fig, width_fraction=1, space_after=12, dpi=600):
        """Add a matplotlib figure as an image to the PDF."""
        available_w, available_h = self.get_page_dimensions()

        # target width
        width = available_w * width_fraction
        height = width * fig.get_size_inches()[1] / fig.get_size_inches()[0]

        # scale down if too tall
        if height > available_h:
            scale = available_h / height
            width *= scale
            height *= scale

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        buf.seek(0)
        img = Image(buf, width=width, height=height)
        self.story.append(img)
        # self.story.append(Spacer(1, space_after))
        plt.close(fig)  # free memory
        
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

    def add_spacer( self, height: int = 12 ):
        """Add a vertical spacer."""
        self.story.append( Spacer( 1, height ) )