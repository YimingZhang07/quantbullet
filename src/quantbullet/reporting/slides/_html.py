"""HTML helpers for slide components: escaping and house-style tables."""

from __future__ import annotations

import datetime as dt
import html
from typing import Sequence

import numpy as np
import pandas as pd

from ..formatters import flex_number_formatter, number2string


def escape(text) -> str:
    """HTML-escape ``text`` (converted with ``str``)."""
    return html.escape(str(text), quote=True)


def _is_missing(value) -> bool:
    return value is None or (pd.api.types.is_scalar(value) and pd.isna(value))


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))


def format_cell(value, fmt=None) -> str:
    """Text for one cell. ``fmt`` is a column format (``ColumnFormat``, ``PdfColumnFormat`` or None).

    Uses the format's ``transformer``, ``formatter`` (PDF formats), ``decimals``, ``comma``,
    ``percent``, ``parens_for_negative`` and ``missing_label`` when present.
    """
    if _is_missing(value):
        label = getattr(fmt, "missing_label", None)
        return "" if label is None else str(label)
    if isinstance(value, (dt.date, pd.Timestamp)):
        return value.strftime("%Y-%m-%d")
    if fmt is None:
        return number2string(value) if _is_number(value) else str(value)
    if fmt.transformer is not None:
        value = fmt.transformer(value)
    formatter = getattr(fmt, "formatter", None)
    if formatter is not None:
        return str(formatter(value))
    if not _is_number(value):
        return str(value)
    parens = getattr(fmt, "parens_for_negative", False) and value < 0
    text = flex_number_formatter(abs(value) if parens else value, decimals=fmt.decimals,
                                 comma=fmt.comma, percent=fmt.percent)
    return f"({text})" if parens else text


def _schema_columns(schema) -> list:
    if schema is None:
        return []
    metas = list(getattr(schema, "columns", schema))
    for meta in metas:
        if not hasattr(meta, "name") or not hasattr(meta, "format"):
            raise TypeError(f"Table schemas hold column metas (ColumnMeta / PdfColumnMeta); got {meta!r}.")
    return metas


def html_table(data: pd.DataFrame, schema=None, *, max_rows: int | None = 25,
               bold_rows: Sequence[int] = (), highlight_neg: bool = True,
               caption: str | None = None, index: bool = False) -> str:
    """A ``<table class="qb-table">`` for ``data``.

    ``schema`` (a ``ColumnSchema`` or list of column metas) picks, orders, labels and
    formats the columns; without it every column is shown with automatic number formats.
    Numeric columns are right-aligned unless the column format sets ``align``.
    ``bold_rows`` are data-row positions (negative counts from the end). ``index=True``
    shows the DataFrame index as the first column.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"A table needs a pandas DataFrame; got {type(data).__name__}.")
    if index:
        data = data.reset_index()
    if max_rows is not None and len(data) > max_rows:
        raise ValueError(f"The table has {len(data)} rows, more than max_rows={max_rows}; "
                         "split it across slides or raise max_rows (layout='dense' fits more).")

    metas = _schema_columns(schema)
    if metas:
        missing = [m.name for m in metas if m.name not in data.columns]
        if missing:
            raise KeyError(f"Schema columns not in the DataFrame: {missing}.")
        columns = [m.name for m in metas]
        headers = [m.display_name or m.name for m in metas]
        formats = [m.format for m in metas]
    else:
        columns = list(data.columns)
        headers = [str(c) for c in columns]
        formats = [None] * len(columns)

    aligns = []
    for column, fmt in zip(columns, formats):
        align = getattr(fmt, "align", None)
        if align:
            aligns.append(align.lower())
        else:
            dtype = data[column].dtype
            numeric = pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
            aligns.append("right" if numeric else "left")

    n = len(data)
    bold = set()
    for row in bold_rows:
        if not -n <= row < n:
            raise IndexError(f"bold_rows position {row} is outside the {n} rows.")
        bold.add(row % n)

    def cell_class(align, extra=()):
        classes = (["num"] if align == "right" else []) + list(extra)
        style = f' style="text-align:{align}"' if align == "center" else ""
        return (f' class="{" ".join(classes)}"' if classes else "") + style

    out = ['<table class="qb-table">']
    if caption:
        out.append(f"<caption>{escape(caption)}</caption>")
    out.append("<thead><tr>" + "".join(f"<th{cell_class(a)}>{escape(h)}</th>"
                                       for h, a in zip(headers, aligns)) + "</tr></thead>")
    out.append("<tbody>")
    for i, row in enumerate(data[columns].itertuples(index=False, name=None)):
        cells = []
        for value, fmt, align in zip(row, formats, aligns):
            raw = value
            if fmt is not None and fmt.transformer is not None and _is_number(raw):
                raw = fmt.transformer(raw)
            negative = highlight_neg and _is_number(raw) and not _is_missing(raw) and raw < 0
            cells.append(f"<td{cell_class(align, ['neg'] if negative else [])}>"
                         f"{escape(format_cell(value, fmt))}</td>")
        row_class = ' class="bold"' if i in bold else ""
        out.append(f"<tr{row_class}>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)
