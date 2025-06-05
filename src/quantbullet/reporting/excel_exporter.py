import os
import pandas as pd
from .columns import ColumnFormat
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter

class ExcelExporter:
    def __init__(self, filename):
        self.filename = filename
        self._default_decimals = 2
        self._default_date_format = "yyyy-mm-dd"
        self._overwrite = False
        self.default_alignment = Alignment(horizontal='right')
        self._sheets = []

    # --- @property + method ---
    @property
    def default_decimals(self):
        return self._default_decimals

    @default_decimals.setter
    def default_decimals(self, value):
        self.set_default_decimals(value)

    def set_default_decimals(self, value: int):
        if not isinstance(value, int) or value < 0:
            raise ValueError("default_decimals must be a non-negative integer.")
        self._default_decimals = value
        return self

    @property
    def default_date_format(self):
        return self._default_date_format

    @default_date_format.setter
    def default_date_format(self, value):
        self.set_default_date_format(value)

    def set_default_date_format(self, value: str):
        """Set the default date format for the Excel file.
        
        The format here has to be compatible with openpyxl. It is not the same as the format in pandas.
        For example, "yyyy-mm-dd" is the correct format for openpyxl, while "%Y-%m-%d" is not.

        Please check https://openpyxl.readthedocs.io/en/3.1.3/_modules/openpyxl/styles/numbers.html.
        """
        if not isinstance(value, str):
            raise ValueError("default_date_format must be a string.")
        self._default_date_format = value
        return self

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

    def add_sheet(self, sheet_name, df, column_formats=None, date_format=None, wrap_header=False):
        
        # Check for duplicate columns
        if self._is_duplicate_columns(df):
            raise ValueError("DataFrame contains duplicate columns. Please rename them before exporting to Excel.")
        
        # keys in column_formats should exist in df.columns
        if column_formats is None:
            column_formats = {}
        for col in column_formats.keys():
            if col not in df.columns:
                raise ValueError(f"Column '{col}' in column_formats does not exist in the DataFrame columns.")

        self._sheets.append({
            "df": df.copy(),
            "sheet_name": sheet_name,
            "column_formats": column_formats or {},
            "date_format": date_format or self._default_date_format,
            "wrap_header": wrap_header
        })
        return self
    
    @staticmethod
    def _is_duplicate_columns(df):
        """Check if the DataFrame has duplicate columns."""
        return df.columns.duplicated().any()
    
    def _build_number_format(self, decimals, fmt: ColumnFormat):
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

    def _get_col_format_strings(self, df, col_formats, date_format):
        """Get the column format strings for the DataFrame."""
        formats = {}
        for col in df.columns:
            fmt = col_formats.get(col, None)
            if pd.api.types.is_numeric_dtype(df[col]):
                decimals = fmt.decimals if fmt and fmt.decimals is not None else self._default_decimals
                pattern = self._build_number_format(decimals, fmt)
                formats[col] = pattern
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                formats[col] = date_format
        return formats

    def _apply_formatting(self, worksheet, df, format_strings, wrap_header=False):
        for idx, col in enumerate(df.columns, 1):
            col_letter = get_column_letter(idx)
            sheet_config = next((s for s in self._sheets if s["df"] is df), None)
            column_format = sheet_config["column_formats"].get(col) if sheet_config else None

            # Set width
            if column_format and column_format.width is not None:
                # HACK It seems there is a gap in column width we set and the actual width exported. and this gap is known to be 0.7.
                width = column_format.width + 0.7
            elif column_format:
                width = column_format.estimate_display_width(df[col], col, self._default_decimals) + 2
            else:
                width = max(df[col].astype(str).map(len).max(), len(col)) + 2  # fallback
            worksheet.column_dimensions[col_letter].width = width

            # Set number format + alignment
            for row in range(2, len(df) + 2):
                cell = worksheet[f"{col_letter}{row}"]
                if col in format_strings:
                    cell.number_format = format_strings[col]
                cell.alignment = self.default_alignment

            # Wrap header text
            header_cell = worksheet[f"{col_letter}1"]
            if wrap_header:
                header_cell.alignment = Alignment(wrap_text=True, horizontal='center', vertical='center')
                
            # Conditional formatting
            if column_format and column_format.color_scale:
                rule = column_format.build_conditional_formatting_rule()
                if rule:
                    cell_range = f"{col_letter}2:{col_letter}{len(df) + 1}"
                    worksheet.conditional_formatting.add(cell_range, rule)
                    
            # Formula Fill
            if column_format and column_format.formula_template:
                for row in range(2, len(df) + 2):
                    formula = column_format.formula_template.format(row=row)
                    worksheet[f"{col_letter}{row}"].value = formula
        if wrap_header:
            worksheet.row_dimensions[1].height = 30
            
    def _apply_column_transforms(self, df, col_formats):
        """Apply transformations to the DataFrame columns based on the provided formats."""
        for col, fmt in col_formats.items():
            if col in df.columns:
                df[col] = fmt.apply_transform(df[col])
        return df

    def save(self):
        """"Save the DataFrames to an Excel file."""
        # HACK the below condition does not make sense, cause if the file exists, it will be overwritten anyway.
        mode = 'w' if self.overwrite else ('a' if os.path.exists(self.filename) else 'w')
        writer_args = {"engine": "openpyxl", "mode": mode}
        if mode == 'a':
            writer_args["if_sheet_exists"] = "replace"

        with pd.ExcelWriter(self.filename, **writer_args) as writer:
            for sheet in self._sheets:
                df = sheet["df"]
                name = sheet["sheet_name"]
                col_formats = sheet["column_formats"]
                # for any transformations, we need to apply them before formatting
                df = self._apply_column_transforms(df, col_formats)
                format_strings = self._get_col_format_strings(df, col_formats, sheet["date_format"])
                df.to_excel(writer, sheet_name=name, index=False)
                worksheet = writer.sheets[name]
                self._apply_formatting(worksheet, df, format_strings, sheet["wrap_header"])

        # self._sheets is a list, so we can clear it
        self._sheets.clear()