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
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .formatters import number2string
from ._reportlab_utils import PdfColumnFormat, PdfColumnMeta, build_table_from_df, multi_index_df_to_table_data, apply_heatmap, make_diverging_colormap

class PdfTextReport:
    def __init__( self, file_path:str, page_size:tuple=None, report_title:str=None, margins:tuple=(36,36,36,36), page_numbering:bool=True ):

        if page_size is None:
            page_size = landscape(letter)
        else:
            page_size = (page_size[0] * inch, page_size[1] * inch)

        # sometimes file_path is a Path object
        if not isinstance(file_path, str):
            file_path = str(file_path)

        self.doc = SimpleDocTemplate(
            file_path,
            pagesize=page_size,
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

    def add_df_table( self, df, schema:list[PdfColumnMeta], space_after:int=12, font_size:int=8 ):
        """Add a DataFrame as a table to the PDF.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to render as a table.
        schema : list of PdfColumnMeta
            Metadata for each column, including formatting and colormap info.
        """
        tbl = build_table_from_df( df, schema )
        # tbl has the style already applied lets append the font size style
        tbl.setStyle( TableStyle([ ("FONTSIZE", (0,0), (-1,-1), font_size) ]) )
        self.story.append( tbl )
        self.story.append( Spacer( 1, space_after ) )

    def add_multiindex_df_table( self, df, font_size:int=6, space_after:int=12, heatmap_cols:bool=False, heatmap_all:bool=False ):
        """Add a MultiIndex DataFrame as a table to the PDF.
        
        This is a less configurable function than the regular add_df_table, due to the complexity of MultiIndex tables.
        This table has focused on displaying the MultiIndex structure clearly. So it is better you have,
            1) your number format ready, can be in strings
        """
        table_data, spans = multi_index_df_to_table_data( df )
        nrow_levels = df.index.nlevels
        ncol_levels = df.columns.nlevels


        heatmap_styles = []
        if heatmap_cols:
            n_rows = len(table_data)
            n_cols = len(table_data[0])
            for col in range(nrow_levels, n_cols):
                _styles = apply_heatmap( table_data=table_data, 
                                         row_range=(nrow_levels, n_rows-1), 
                                         col_range=(col, col),
                                         cmap=make_diverging_colormap(),
                                         vmid=0 )
                heatmap_styles.extend(_styles)

        if heatmap_all:
            # all the cells color as a whole
            n_rows = len(table_data)
            n_cols = len(table_data[0])
            _styles = apply_heatmap( table_data=table_data, 
                                            row_range=(nrow_levels, n_rows-1),
                                            col_range=(nrow_levels, n_cols-1),
                                            cmap=make_diverging_colormap( high_color="#63be7b", mid_color=(1,1,1), low_color="#f8696b" ),
                                            vmid=10 )
            heatmap_styles.extend(_styles)

        main_styles = [
            ("GRID", (0, 0), (-1, -1), 0.5, "black"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (0, -1), "MIDDLE"),
            ("ALIGN", (nrow_levels, ncol_levels), (-1, -1), "RIGHT"),
            ("BACKGROUND", (0, 0), (-1, ncol_levels - 1), "#d9d9d9"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ]
        extended_styles = main_styles + spans + heatmap_styles

        table_style = TableStyle( extended_styles )

        tbl = Table(table_data, style=table_style)

        self.story.append( tbl )
        self.story.append( Spacer( 1, space_after ) )

    def add_text( self, text:str, font_size:int=10, space_after:int=12 ):
        """Add a left-aligned text paragraph."""
        style = ParagraphStyle(
            name="NormalStyle",
            fontName="Helvetica",
            fontSize=font_size,
            alignment=0  # left
        )
        p = Paragraph(text, style=style)
        self.story.append(p)
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
        text = f"{page_num}"
        width, height = self.doc.pagesize
        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(width / 2.0, 15, text)  # y=15 points from bottom