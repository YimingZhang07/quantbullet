import os
import numpy as np
import pandas as pd
from quantbullet.core.enums import DataType
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule
from typing import List, Dict, Callable, Optional


def df_to_html_table(df: pd.DataFrame, table_title: str = '') -> str:
    """
    Function to convert a pandas DataFrame into an HTML table.

    Parameters
    ----------
    df : pd.DataFrame
    """

    # Turn the columns that are floats into strings with 2 decimal places
    for col in df.select_dtypes(include='float').columns:
        df.loc[:, col] = df[col].apply(lambda x: f'{x:.2f}')

    return f"""
    <div class="table-title">{table_title}</div>
    <table>
        <thead>
            <tr>{''.join(f'<th>{col}</th>' for col in df.columns)}</tr>
        </thead>
        <tbody>
            {''.join(f'<tr>{"".join(f"<td>{value}</td>" for value in row)}</tr>' for row in df.values)}
        </tbody>
    </table>
    """

# Function to generate the complete HTML page
def generate_html_page(body_content: str, table_col_width = "20%", table_width = "80%") -> str:

    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            <!-- h2 {{ color: #4CAF50; text-align: center; }} -->
            table {{ width: {table_width}; border-collapse: collapse; margin: 10px 0; }}
            th {{ background-color: #005CAF; color: white; font-weight: bold; padding: 2px; text-align: right; width: {table_col_width}; }}
            td {{ padding: 2px; border-bottom: 1px solid #ddd; text-align: right; width: {table_col_width}; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .table-title {{ font-weight: normal; text-align: left; }}
        </style>
    </head>
    <body>
        {body_content}
    </body>
    </html>
    """

def consolidate_data_types(df: pd.DataFrame, type_map: dict, replace_with_none : bool = False) -> pd.DataFrame:
    """
    Consolidates the data types in the DataFrame based on the provided type map.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be consolidated.
    type_map : dict
        A dictionary mapping column names to their respective data types.
    replace_with_None : bool
        If True, replace invalid conversions with None instead of NaN.

    Returns
    -------
    pd.DataFrame
        The DataFrame with consolidated data types.
    """
    for column, dtype in type_map.items():
        if column in df.columns:
            if dtype == DataType.DATE:
                df[column] = pd.to_datetime(df[column], errors='coerce').dt.date
            elif dtype == DataType.STRING:
                df[column] = df[column].astype(str)
            elif dtype == DataType.FLOAT:
                df[column] = pd.to_numeric(df[column], errors='coerce').astype(float) # consider to use 'float64'
            elif dtype == DataType.INT:
                df[column] = pd.to_numeric(df[column], errors='coerce').astype(int) # consider to use 'Int64'
            elif dtype == DataType.BOOL:
                df[column] = df[column].astype(bool)
            elif dtype == DataType.DATETIME:
                df[column] = pd.to_datetime(df[column], errors='coerce')

    if replace_with_none:
        df.replace( { np.nan: None, pd.NaT: None }, inplace=True )
    
    return df


def combine_latest_records(dataframes: List[pd.DataFrame], unique_keys: List[str]) -> pd.DataFrame:
    """
    Combines a sequence of DataFrames, keeping only the latest records based on unique keys.
    
    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        A list of DataFrames to be combined.
    unique_keys : List[str]
        A list of columns that uniquely identify each record.

    Returns
    -------
    pd.DataFrame
        The combined DataFrame containing the latest records.
    """
    if not dataframes:
        raise ValueError("The dataframes list cannot be empty.")
    if not unique_keys:
        raise ValueError("The unique_keys list cannot be empty.")
    
    # Initialize an empty master dataframe with columns from the first dataframe
    master_df = pd.DataFrame(columns=dataframes[0].columns)
    
    # Iterate over the list of dataframes
    for df in dataframes:
        if not all(key in df.columns for key in unique_keys):
            raise ValueError(f"All unique_keys must be present in each DataFrame. Missing in {df.columns}")
        
        # Combine and drop duplicates based on unique keys
        master_df = pd.concat([master_df, df]).drop_duplicates(subset=unique_keys, keep='last')
    
    # Reset the index before returning
    master_df = master_df.reset_index(drop=True)
    return master_df

class ColumnFormat:
    def __init__(
        self,
        decimals=None,
        width=None,
        percent=False,
        comma=False,
        parens_for_negative=False,
        transform=None,
        color_scale=None,
        higher_is_better=True,
        formula_template: Optional[str] = None
    ):
        self.decimals = decimals
        self.width = width
        self.percent = percent
        self.comma = comma
        self.parens_for_negative = parens_for_negative
        self.transform = transform  # e.g., lambda x: x * 100
        self.color_scale = color_scale
        self.higher_is_better = higher_is_better
        self.formula_template = formula_template  # e.g., "=SUM(A1:A10)" for Excel formulas

    def apply_transform(self, series):
        if self.transform:
            return series.apply(self.transform)
        return series
    
    def build_conditional_formatting_rule(self):
        if self.color_scale and self.higher_is_better:
            return ColorScaleRule(
                start_type='max', start_color='63BE7B',  # green
                mid_type='percentile', mid_value=50, mid_color='FFFFFF',  # white
                end_type='min', end_color='F8696B'  # red
            )
        elif self.color_scale and not self.higher_is_better:
            return ColorScaleRule(
                start_type='min', start_color='F8696B',  # red
                mid_type='percentile', mid_value=50, mid_color='FFFFFF',
                end_type='max', end_color='63BE7B'  # green
            )
        return None
    
    def estimate_display_width(self, series: pd.Series, header: str, default_decimals: int = 2):
        decimals = self.decimals if self.decimals is not None else default_decimals
        use_comma = self.comma
        use_parens = self.parens_for_negative
        use_percent = self.percent

        # Apply transformation
        series = self.apply_transform(series)

        def preview(val):
            if pd.isna(val):
                return ""
            try:
                v = round(val, decimals)
                s = f"{v:,.{decimals}f}" if use_comma else f"{v:.{decimals}f}"
                if use_parens and v < 0:
                    s = f"({s.strip('-')})"
                if use_percent:
                    s += "%"
                return s
            except Exception:
                return str(val)

        max_data_len = series.map(preview).map(len).max()
        return max(max_data_len, len(header))

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


class HTMLTableBuilder:
    def __init__(
        self,
        df: pd.DataFrame,
        title: str = '',
        column_settings: Optional[Dict[str, Dict]] = None,
        css_class: str = 'default-table',
        inline_style: bool = True,
        table_width: str = '80%',
        custom_css: Optional[str] = None,
    ):
        """Initialize the HTMLTableBuilder with a DataFrame and optional settings.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be converted to HTML.
        title : str, optional
            The title of the table, displayed above it.
        column_settings : dict, optional
            A dictionary containing settings for each column; accept keys like 'width', 'align', and 'formatter'.
        css_class : str, optional
            The CSS class to apply to the table.
        inline_style : bool, optional
            If True, apply inline styles to the table and its elements. This needs to be True if you want the column settings to be applied.
        table_width : str, optional
            The width of the table, default is '80%'.
            Note that the column width will be relative to this width.
        custom_css : str, optional
            Custom CSS styles to be applied to the table. This is used to override the default styles in a HTMLPageBuilder.
        """
        self.df                 = df.copy()
        self.title              = title
        self.column_settings    = column_settings or {}
        self.css_class          = css_class
        self.inline_style       = inline_style
        self.table_width        = table_width
        self.custom_css         = custom_css
        
    @property
    def _default_column_width(self):
        n_cols = len(self.df.columns)
        return f'{100 / n_cols:.2f}%'

    def format_cell(self, col, val):
        fmt: Optional[Callable] = self.column_settings.get(col, {}).get('formatter')
        return fmt(val) if fmt else val

    def generate_colgroup(self):
        colgroup_html = '<colgroup>'
        for col in self.df.columns:
            width = self.column_settings.get(col, {}).get('width', self._default_column_width)
            colgroup_html += f'<col style="width: {width};">'
        colgroup_html += '</colgroup>'
        return colgroup_html

    def generate_header(self):
        headers = ''
        for col in self.df.columns:
            align = self.column_settings.get(col, {}).get('align', 'right')
            style = f'text-align: {align};' if self.inline_style else ''
            headers += f'<th style="{style}">{col}</th>'
        return headers

    def generate_rows(self):
        rows = ''
        for _, row in self.df.iterrows():
            row_html = ''
            for col in self.df.columns:
                val = self.format_cell(col, row[col])
                align = self.column_settings.get(col, {}).get('align', 'right')
                style = f'text-align: {align};' if self.inline_style else ''
                row_html += f'<td style="{style}">{val}</td>'
            rows += f'<tr>{row_html}</tr>'
        return rows

    def to_html(self):
        table_style = f'style="width: {self.table_width}; border-collapse: collapse; margin: 10px 0;"' if self.inline_style else ''
        scoped_style = f"<style>{self.custom_css}</style>" if self.custom_css else ''
        return f"""
        <div class="table-title">{self.title}</div>
        {scoped_style}
        <table class="{self.css_class}" {table_style}>
            {self.generate_colgroup()}
            <thead><tr>{self.generate_header()}</tr></thead>
            <tbody>{self.generate_rows()}</tbody>
        </table>
        """

class HTMLPageBuilder:
    def __init__(self, global_styles: Optional[str] = None):
        self.elements: List[str] = []
        self.global_styles = global_styles or self.default_global_styles

    @property
    def default_global_styles(self):
        return """
        body { font-family: Arial, sans-serif; padding: 20px; }
        h1, h2 { color: #333; }
        p { font-size: 14px; line-height: 1.6; }
        .default-table { border-collapse: collapse; margin: 20px 0; width: 80%; }
        .default-table th, .default-table td { padding: 6px 8px; border: 1px solid #ddd; }
        .default-table tr:nth-child(even) { background-color: #f9f9f9; }
        .default-table tr:hover { background-color: #f1f1f1; }
        .table-title { font-weight: bold; margin-top: 20px; }
        """

    def add_paragraph(self, text: str):
        self.elements.append(f"<p>{text}</p>")

    def add_heading(self, text: str, level: int = 2):
        self.elements.append(f"<h{level}>{text}</h{level}>")

    def add_raw_html(self, html: str):
        self.elements.append(html)

    def add_table(self, html_table: str):
        self.elements.append(html_table)

    def build(self) -> str:
        body_content = '\n'.join(self.elements)
        return f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>{self.global_styles}</style>
        </head>
        <body>
            {body_content}
        </body>
        </html>
        """
