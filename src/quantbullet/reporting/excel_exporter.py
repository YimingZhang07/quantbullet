import io
import os
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import coordinate_from_string
from .columns import ColumnFormat, ColumnSchema

# Excel's default font size; column widths are measured in characters of this font
_DEFAULT_FONT_SIZE = 11


@dataclass
class _TableBlock:
    """One DataFrame queued for a sheet, at its resolved position."""
    sheet_name: str
    df: pd.DataFrame
    column_formats: Dict[str, ColumnFormat]
    start_row: int
    start_col: int
    wrap_header: bool
    include_header: bool

    @property
    def last_row(self) -> int:
        return self.start_row + len(self.df) + (1 if self.include_header else 0) - 1


class ExcelExporter:
    def __init__(self, filename):
        self.filename = filename
        self._overwrite = False
        self.default_alignment = Alignment(horizontal='right')
        self._blocks = []
        self._freeze_panes = {}
        self._font = {}

    @property
    def overwrite(self):
        return self._overwrite

    @overwrite.setter
    def overwrite(self, value):
        self.set_overwrite(value)

    def set_overwrite(self, value: bool):
        """Set whether to overwrite the existing Excel file or not.

        If you want to reuse the same file, just write to different sheets, then set this to False.
        """
        self._overwrite = bool(value)
        return self

    def set_freeze_panes(self, sheet_name, cell: Optional[str]):
        """Freeze the rows above and the columns left of ``cell`` on ``sheet_name``.

        For example "A2" freezes the first row, "B2" the first row and column A. ``None`` unfreezes the sheet.
        """
        if cell is not None:
            try:
                coordinate_from_string(cell)
            except Exception as e:
                raise ValueError(f"Invalid cell '{cell}' for freeze panes; use a cell such as 'B2'.") from e
        self._freeze_panes[sheet_name] = cell
        return self

    def set_font(self, name: Optional[str] = None, size: Optional[float] = None, color: Optional[str] = None):
        """Set the font of every cell the tables write, on all sheets; headers stay bold.

        ``name`` is a font family such as "Arial", ``size`` in points and ``color`` a hex RGB such as "1F4E79".
        ``None`` keeps Excel's default (Calibri 11, black). Estimated column widths and the wrapped header
        height scale with ``size``.
        """
        self._font = {key: value for key, value in dict(name=name, size=size, color=color).items() if value is not None}
        return self

    @property
    def _font_scale(self):
        return self._font.get("size", _DEFAULT_FONT_SIZE) / _DEFAULT_FONT_SIZE

    def add_sheet(self, sheet_name, df, schema: Optional[ColumnSchema] = None, start_col=1, start_row=None,
                  wrap_header=False, include_header=True, gap_rows=1):
        """Queue ``df`` to be written to ``sheet_name``.

        Several calls with the same ``sheet_name`` put several tables on one sheet. With ``start_row=None``
        the first table on a sheet starts at row 1 and each later one goes below the tables already on it,
        ``gap_rows`` blank rows apart.

        With a ``schema``, columns are renamed to their display names and only the schema's columns are
        written, in schema order.
        """
        if schema:
            missing = [col.name for col in schema.columns if col.name not in df.columns]
            if missing:
                raise ValueError(f"Schema columns {missing} do not exist in the DataFrame columns.")
            df = df.rename(columns=schema.rename_map)[schema.display_order]
            column_formats = dict(schema.col_config)
        else:
            df = df.copy()
            column_formats = {col: ColumnFormat() for col in df.columns}

        if self._is_duplicate_columns(df):
            raise ValueError("DataFrame contains duplicate columns. Please rename them before exporting to Excel.")

        if start_row is None:
            last_rows = [block.last_row for block in self._blocks if block.sheet_name == sheet_name]
            start_row = max(last_rows) + gap_rows + 1 if last_rows else 1

        self._blocks.append(_TableBlock(
            sheet_name=sheet_name,
            df=df,
            column_formats=column_formats,
            start_row=start_row,
            start_col=start_col,
            wrap_header=wrap_header,
            include_header=include_header,
        ))
        return self

    @staticmethod
    def _is_duplicate_columns(df):
        """Check if the DataFrame has duplicate columns."""
        return df.columns.duplicated().any()

    def _build_number_format(self, decimals, fmt: ColumnFormat):
        """Build the Excel number format; ``None`` for ``decimals=None``, which leaves cells in General."""
        if decimals is None:
            return None

        base = '0' + ('.' + '0' * decimals if decimals > 0 else '')

        if fmt is None:
            return base

        if fmt.comma:
            base = "#,##" + base

        if fmt.percent:
            base += "%"

        # Add negative in parentheses
        if fmt.parens_for_negative:
            # Excel format: positive;negative;zero
            base = f"{base};({base});{base}"

        return base

    def _get_col_format_strings(self, df, col_formats):
        """Get the column format strings for the DataFrame."""
        formats = {}
        for col in df.columns:
            fmt = col_formats[col]
            if pd.api.types.is_numeric_dtype(df[col]):
                formats[col] = self._build_number_format(fmt.decimals, fmt)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                formats[col] = fmt.date_format
        return formats

    def _apply_formatting(self, worksheet, df, block: _TableBlock, widths):
        """Format the cells of one written table; column widths are collected into ``widths``."""
        format_strings = self._get_col_format_strings(df, block.column_formats)
        data_start_row = block.start_row + 1 if block.include_header else block.start_row
        data_font = Font(**self._font) if self._font else None
        header_font = Font(bold=True, **self._font) if self._font else None
        for j, col in enumerate(df.columns):
            col_index = block.start_col + j
            col_letter = get_column_letter(col_index)
            column_format = block.column_formats[col]

            # Width: tables sharing a column on one sheet get the widest of their widths
            if column_format.width is not None:
                # HACK It seems there is a gap in column width we set and the actual width exported. and this gap is known to be 0.7.
                width = column_format.width + 0.7
            else:
                width = (column_format.estimate_display_width(df[col], str(col)) + 2) * self._font_scale
            key = (block.sheet_name, col_letter)
            widths[key] = max(widths.get(key, 0), width)

            # Format data cells
            for i in range(len(df)):
                cell = worksheet.cell(row=data_start_row + i, column=col_index)
                cell.alignment = self.default_alignment
                if data_font:
                    cell.font = data_font
                if format_strings.get(col):
                    cell.number_format = format_strings[col]

            if header_font and block.include_header:
                worksheet.cell(row=block.start_row, column=col_index).font = header_font

            # Wrap header text
            if block.wrap_header and block.include_header:
                header_cell = worksheet.cell(row=block.start_row, column=col_index)
                header_cell.alignment = Alignment(wrap_text=True, horizontal='center', vertical='center')
                worksheet.row_dimensions[block.start_row].height = 30 * self._font_scale

            # Conditional formatting
            if column_format.color_scale and len(df):
                rule = column_format.build_conditional_formatting_rule()
                if rule:
                    cell_range = f"{col_letter}{data_start_row}:{col_letter}{data_start_row + len(df)-1}"
                    worksheet.conditional_formatting.add(cell_range, rule)

            # Formula Fill
            if column_format.formula_template:
                for i in range(len(df)):
                    formula = column_format.formula_template.format(row=data_start_row + i)
                    worksheet.cell(row=data_start_row + i, column=col_index).value = formula

    def _apply_column_transforms(self, df, col_formats):
        """Apply transformations to the DataFrame columns based on the provided formats."""
        for col, fmt in col_formats.items():
            if col in df.columns:
                df[col] = fmt.apply_transform(df[col])
        return df

    def _clear_sheets(self, book):
        """Replace each queued sheet that already exists with an empty one at the same position."""
        for name in dict.fromkeys(block.sheet_name for block in self._blocks):
            if name in book.sheetnames:
                index = book.sheetnames.index(name)
                book.remove(book[name])
                book.create_sheet(name, index)

    def save(self):
        """Write the queued tables to the Excel file.

        Each sheet named in the queue is cleared and rewritten with all of its tables. The other sheets of an
        existing file are kept, unless ``overwrite`` is set. The workbook is built in memory and the file is
        written only after every table succeeded, so on failure the file is untouched and the queue is kept.
        """
        if not self._blocks:
            raise ValueError("No tables to save. Call add_sheet first.")
        unknown = set(self._freeze_panes) - {block.sheet_name for block in self._blocks}
        if unknown:
            raise ValueError(f"Freeze panes set for sheets without tables: {sorted(unknown)}.")

        if not self.overwrite and os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                buffer = io.BytesIO(f.read())
            writer = pd.ExcelWriter(buffer, engine="openpyxl", mode="a", if_sheet_exists="overlay")
        else:
            buffer = io.BytesIO()
            writer = pd.ExcelWriter(buffer, engine="openpyxl", mode="w")

        self._clear_sheets(writer.book)
        widths = {}
        for block in self._blocks:
            # for any transformations, we need to apply them before formatting
            df = self._apply_column_transforms(block.df.copy(), block.column_formats)
            df.to_excel(writer, sheet_name=block.sheet_name, index=False, header=block.include_header,
                        startrow=block.start_row - 1, startcol=block.start_col - 1)
            self._apply_formatting(writer.sheets[block.sheet_name], df, block, widths)
        for (sheet_name, col_letter), width in widths.items():
            writer.sheets[sheet_name].column_dimensions[col_letter].width = width
        for sheet_name, cell in self._freeze_panes.items():
            writer.sheets[sheet_name].freeze_panes = cell
        writer.close()

        with open(self.filename, "wb") as f:
            f.write(buffer.getvalue())
        # The queue and the freeze panes belong to this save; overwrite and the font stay for the next one
        self._blocks.clear()
        self._freeze_panes.clear()
