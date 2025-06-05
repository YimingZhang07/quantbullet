import pandas as pd
from typing import List, Dict, Callable, Optional

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