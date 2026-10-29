import re
import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment
from typing import Optional
from dataclasses import dataclass, field
from typing import List, Dict, Callable
from .base import BaseColumnFormat, BaseColumnMeta
from .formatters import flex_number_formatter

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

@dataclass
class ColumnFormat( BaseColumnFormat ):
    """How one column is written to Excel; ``decimals``, ``comma``, ``transformer`` and ``percent`` come from the base."""
    width               : Optional[ float ] = None    # fixed width in characters; None fits values and header
    parens_for_negative : bool              = False
    color_scale         : bool              = False
    higher_is_better    : bool              = True
    # An Excel format code, not a strftime one: "yyyy-mm-dd", not "%Y-%m-%d"; None means "yyyy-mm-dd"
    date_format         : Optional[ str ]   = "yyyy-mm-dd"
    formula_template    : Optional[ str ]   = None    # e.g. "=D{row}*E{row}", filled per data row
    # Horizontal alignment of the data cells, whatever their type: "left", "center", "right" or another openpyxl
    # value; None uses the exporter's default
    align               : Optional[ str ]   = None

    def __post_init__(self):
        if self.date_format is None:
            self.date_format = "yyyy-mm-dd"
        Alignment(horizontal=self.align)   # raises ValueError for an unknown alignment

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
        is_date = pd.api.types.is_datetime64_any_dtype(series)

        def preview(val):
            if pd.isna(val):
                return ""
            if is_date:
                return _excel_date_text(val, self.date_format)
            if self.decimals is None:
                # General shows floats with up to about 10 significant digits
                return f"{val:.10g}" if isinstance(val, float) else str(val)
            try:
                text = flex_number_formatter(val, decimals=self.decimals, comma=self.comma, percent=self.percent)
            except (TypeError, ValueError):
                return str(val)
            return f"({text.lstrip('-')})" if self.parens_for_negative and val < 0 else text

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