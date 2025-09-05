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
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image
from .formatters import number2string
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

class PdfTextReport:
    def __init__( self, file_path:str, page_size:str=None, report_title:str=None ):
        self.doc = SimpleDocTemplate(
            file_path,
            pagesize=landscape(letter)
        )
        self.story = []
        self.report_title = report_title
        self.add_centered_text( report_title, font_size=16, space_after=24 )
        
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
        self.doc.build( self.story )