from ._reportlab_utils import PdfColumnMeta
from .columns import ColumnFormat, ColumnMeta, ColumnSchema


def convert_pdf_schema_to_excel_schema(pdf_schema: list[PdfColumnMeta]) -> ColumnSchema:
    """Convert a list of PdfColumnMeta into an Excel-compatible ColumnSchema."""
    columns = []
    for col in pdf_schema:
        excel_fmt = ColumnFormat(
            decimals=col.format.decimals,
            comma=col.format.comma,
            percent=col.format.percent,
            transformer=col.format.transformer,
            color_scale=col.format.colormap is not None,
        )
        columns.append(ColumnMeta(name=col.name, display_name=col.display_name, format=excel_fmt))
    return ColumnSchema(columns=columns)
