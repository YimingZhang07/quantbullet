from ._reportlab_utils import PdfColumnMeta, _flatten_schema
from .columns import ColumnFormat, ColumnMeta, ColumnSchema


def convert_pdf_schema_to_excel_schema(pdf_schema) -> ColumnSchema:
    """Convert a list of PdfColumnMeta (or PdfColumnGroup) into an Excel-compatible ColumnSchema."""
    flat_cols, _ = _flatten_schema(pdf_schema)
    columns = []
    for col in flat_cols:
        excel_fmt = ColumnFormat(
            decimals=col.format.decimals,
            comma=col.format.comma,
            percent=col.format.percent,
            transformer=col.format.transformer,
            color_scale=col.format.colormap is not None,
        )
        columns.append(ColumnMeta(name=col.name, display_name=col.display_name, format=excel_fmt))
    return ColumnSchema(columns=columns)
