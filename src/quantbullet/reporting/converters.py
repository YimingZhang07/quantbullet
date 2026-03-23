from ._reportlab_utils import PdfColumnGroup, PdfColumnMeta, _flatten_schema
from .columns import ColumnFormat, ColumnMeta, ColumnSchema


def convert_pdf_schema_to_excel_schema(pdf_schema) -> ColumnSchema:
    """Convert a list of PdfColumnMeta (or PdfColumnGroup) into an Excel-compatible ColumnSchema.

    For grouped schemas, display names are prefixed with the group label
    (e.g. "DM" -> "Prod DM") to avoid duplicate column names in Excel.
    """
    flat_cols, group_spans = _flatten_schema(pdf_schema)

    col_to_group = {}
    for start, end, label, _, _ in group_spans:
        for i in range(start, end + 1):
            col_to_group[i] = label

    columns = []
    for idx, col in enumerate(flat_cols):
        display = col.display_name or col.name
        group_label = col_to_group.get(idx)
        if group_label and display != group_label:
            display = f"{group_label} {display}"

        excel_fmt = ColumnFormat(
            decimals=col.format.decimals,
            comma=col.format.comma,
            percent=col.format.percent,
            transformer=col.format.transformer,
            color_scale=col.format.colormap is not None,
        )
        columns.append(ColumnMeta(name=col.name, display_name=display, format=excel_fmt))
    return ColumnSchema(columns=columns)
