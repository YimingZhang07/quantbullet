import re
import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from typing import Optional
from dataclasses import dataclass, field
from typing import List, Dict, Callable
from .base import BaseColumnFormat, BaseColumnMeta

_MONTHS = ("January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
# Quoted text, an escaped character, a [bracketed] prefix, AM/PM, a run of one date/time code letter, or any character
_DATE_TOKEN = re.compile(r'"[^"]*"|\\.|\[[^\]]*\]|AM/PM|A/P|y+|m+|d+|h+|s+|.', re.IGNORECASE)


def _excel_date_text(value, date_format: str) -> str:
    """Render ``value`` roughly as Excel shows it with the Excel ``date_format``; used to estimate column widths.

    ``m`` and ``mm`` are minutes right after an hour code or right before a seconds code, as in Excel.
    Month and weekday names are English.
    """
    tokens = _DATE_TOKEN.findall(date_format)
    codes = [token.lower()[0] if token.lower()[0] in "ymdhs" and len(set(token.lower())) == 1 else None
             for token in tokens]
    twelve_hour = any(token.upper() in ("AM/PM", "A/P") for token in tokens)
    hour = (value.hour % 12 or 12) if twelve_hour else value.hour

    parts = []
    for i, token in enumerate(tokens):
        code, n = codes[i], len(token)
        if code == "y":
            parts.append(f"{value.year:04d}" if n > 2 else f"{value.year % 100:02d}")
        elif code == "m":
            previous = next((c for c in reversed(codes[:i]) if c), None)
            following = next((c for c in codes[i + 1:] if c), None)
            if n <= 2 and (previous == "h" or following == "s"):
                parts.append(f"{value.minute:0{n}d}")
            else:
                month = _MONTHS[value.month - 1]
                parts.append({1: str(value.month), 2: f"{value.month:02d}", 3: month[:3], 4: month}.get(n, month[0]))
        elif code == "d":
            weekday = _WEEKDAYS[value.weekday()]
            parts.append({1: str(value.day), 2: f"{value.day:02d}", 3: weekday[:3]}.get(n, weekday))
        elif code == "h":
            parts.append(f"{hour:0{min(n, 2)}d}")
        elif code == "s":
            parts.append(f"{value.second:0{min(n, 2)}d}")
        elif token.upper() == "AM/PM":
            parts.append("AM" if value.hour < 12 else "PM")
        elif token.upper() == "A/P":
            parts.append("A" if value.hour < 12 else "P")
        elif token.startswith('"'):
            parts.append(token[1:-1])
        elif token.startswith("\\"):
            parts.append(token[1:])
        elif not token.startswith("["):
            parts.append(token)
    return "".join(parts)

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
    
    def estimate_display_width(self, series: pd.Series, header: str):
        """Estimate the display width of a column from its values and header.

        ``series`` holds the values as written to the sheet, i.e. after ``apply_transform``.
        With ``decimals=None`` the values are shown in Excel's General format, about 10 significant digits.
        Dates are rendered with ``date_format``.
        """
        decimals = self.decimals
        use_comma = self.comma
        use_parens = self.parens_for_negative
        use_percent = self.percent
        is_date = pd.api.types.is_datetime64_any_dtype(series)

        def preview(val):
            if pd.isna(val):
                return ""
            if is_date:
                return _excel_date_text(val, self.date_format)
            if decimals is None:
                # General shows floats with up to about 10 significant digits
                return f"{val:.10g}" if isinstance(val, float) else str(val)
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

        max_data_len = int(series.map(preview).map(len).max()) if len(series) else 0
        return max(max_data_len, len(header))

@dataclass
class ColumnMeta( BaseColumnMeta ):
    # override the format type
    format: ColumnFormat = field( default_factory=ColumnFormat )

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