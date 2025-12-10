import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from typing import Optional
from dataclasses import dataclass, field
from typing import List, Dict, Callable
from .base import BaseColumnFormat, BaseColumnMeta

class ColumnFormat( BaseColumnFormat ):
    def __init__(
        self,
        decimals=2,
        width=None,
        percent=False,
        comma=False,
        parens_for_negative=False,
        transformer=None,
        color_scale=None,
        higher_is_better=True,
        date_format: Optional[str]=None,
        formula_template: Optional[str] = None
    ):
        super().__init__(
            decimals=decimals,
            percent=percent,
            comma=comma,
            transformer=transformer
        )
        self.width = width
        self.parens_for_negative = parens_for_negative
        self.color_scale = color_scale
        self.higher_is_better = higher_is_better
        self.date_format = date_format or self._default_date_format
        self.formula_template = formula_template  # e.g., "=SUM(A1:A10)" for Excel formulas
        
    @property
    def _default_date_format(self):
        """Set the default date format for the Excel file.
        
        The format here has to be compatible with openpyxl. It is not the same as the format in pandas.
        For example, "yyyy-mm-dd" is the correct format for openpyxl, while "%Y-%m-%d" is not.

        Please check https://openpyxl.readthedocs.io/en/3.1.3/_modules/openpyxl/styles/numbers.html.
        """
        return "yyyy-mm-dd"

    def apply_transform(self, series):
        if self.transformer:
            return series.apply(self.transformer)
        return series
    
    def build_conditional_formatting_rule(self):
        if self.color_scale and not self.higher_is_better:
            return ColorScaleRule(
                start_type='min', start_color='63BE7B',  # green
                mid_type='percentile', mid_value=50, mid_color='FFFFFF',  # white
                end_type='max', end_color='F8696B'  # red
            )
        elif self.color_scale and self.higher_is_better:
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
    
@dataclass
class ColumnMeta( BaseColumnMeta ):
    pass

# A utility to manage the whole config
class ColumnSchema:
    def __init__(self, columns: List[ColumnMeta]):
        self.columns = columns

    @property
    def rename_map(self) -> Dict[str, str]:
        return {col.name: col.display_name for col in self.columns}

    @property
    def col_config(self) -> Dict[str, ColumnFormat]:
        return {col.display_name: col.format for col in self.columns}

    @property
    def display_order(self) -> List[str]:
        return [col.display_name for col in self.columns]

    def filter(self, predicate: Callable[[ColumnMeta], bool]) -> 'ColumnSchema':
        return ColumnSchema([col for col in self.columns if predicate(col)])