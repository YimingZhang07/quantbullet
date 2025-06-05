import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from typing import Optional
from dataclasses import dataclass, field
from typing import List, Dict, Callable

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
class ColumnMeta:
    raw_name: str
    display_name: Optional[str] = None
    format: ColumnFormat = field(default_factory=ColumnFormat)

    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.raw_name

# A utility to manage the whole config
class ColumnSchema:
    def __init__(self, columns: List[ColumnMeta]):
        self.columns = columns

    @property
    def rename_map(self) -> Dict[str, str]:
        return {col.raw_name: col.display_name for col in self.columns}

    @property
    def col_config(self) -> Dict[str, ColumnFormat]:
        return {col.display_name: col.format for col in self.columns}

    @property
    def display_order(self) -> List[str]:
        return [col.display_name for col in self.columns]

    def filter(self, predicate: Callable[[ColumnMeta], bool]) -> 'ColumnSchema':
        return ColumnSchema([col for col in self.columns if predicate(col)])