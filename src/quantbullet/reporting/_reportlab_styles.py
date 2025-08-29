from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

class AdobeSourceFontStyles:
    # Sans-serif (Source Sans 3) → titles, subtitles
    SansTitle = ParagraphStyle(
        "SansTitle",
        fontName="SourceSans3-Regular",
        fontSize=32,
        alignment=TA_CENTER,
        spaceAfter=30,
    )

    SansSubtitle = ParagraphStyle(
        "SansSubtitle",
        fontName="SourceSans3-Regular",
        fontSize=18,
        alignment=TA_CENTER,
        textColor="gray",
        spaceAfter=40,
    )

    # Serif (Source Serif 4) → body text
    SerifNormal = ParagraphStyle(
        "SerifNormal",
        fontName="SourceSerif4-Regular",
        fontSize=12,
        alignment=TA_LEFT,
        leading=18,
        spaceAfter=12,
    )

    SerifBold = ParagraphStyle(
        "SerifBold",
        fontName="SourceSerif4-Bold",
        fontSize=12,
        alignment=TA_LEFT,
        leading=18,
        spaceAfter=12,
    )

    SerifItalic = ParagraphStyle(
        "SerifItalic",
        fontName="SourceSerif4-Italic",
        fontSize=12,
        alignment=TA_LEFT,
        leading=18,
        spaceAfter=12,
    )

    # Monospace (Source Code Pro) → code, tables, numbers
    MonoCode = ParagraphStyle(
        "MonoCode",
        fontName="SourceCodePro-Regular",
        fontSize=10,
        alignment=TA_LEFT,
        leading=16,
        spaceAfter=12,
    )
