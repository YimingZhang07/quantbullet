import os
import numpy as np
import pandas as pd
from quantbullet.core.enums import DataType
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
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

    def add_sheet(self, df, sheet_name, round_cols=None, date_format=None):
        self._sheets.append({
            "df": df.copy(),
            "sheet_name": sheet_name,
            "round_cols": round_cols or {},
            "date_format": date_format or self._default_date_format
        })
        return self

    def _get_col_formats(self, df, round_cols, date_format):
        """Get the column formats for the DataFrame."""
        formats = {}
        for col in df.select_dtypes(include='number').columns:
            decimals = round_cols.get(col, self.default_decimals)
            formats[col] = '0' if decimals == 0 else f'0.{"0" * decimals}'
        for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns:
            formats[col] = date_format
        return formats

    def _apply_formatting(self, worksheet, df, col_formats):
        """Apply formatting to the worksheet based on the DataFrame."""
        for idx, col in enumerate(df.columns, 1):
            col_letter = get_column_letter(idx)
            max_len = max(df[col].astype(str).map(len).max(), len(str(col))) + 8
            worksheet.column_dimensions[col_letter].width = max_len

            for row in range(2, len(df) + 2):
                cell = worksheet[f"{col_letter}{row}"]
                if col in col_formats:
                    cell.number_format = col_formats[col]
                cell.alignment = self.default_alignment

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
                formats = self._get_col_formats(df, sheet["round_cols"], sheet["date_format"])
                df.to_excel(writer, sheet_name=name, index=False)
                worksheet = writer.sheets[name]
                self._apply_formatting(worksheet, df, formats)

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
