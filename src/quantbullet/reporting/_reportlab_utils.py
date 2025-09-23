from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from .formatters import default_number_formatter
from .base import BaseColumnFormat, BaseColumnMeta

def hex_to_rgb01(hex_str: str):
    """Convert a hex color string to an RGB tuple with values between 0 and 1."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def make_diverging_colormap(low_color=(0, 1, 0), mid_color=(1, 1, 1), high_color=(1, 0, 0)):
    """Create a diverging colormap function that maps values to colors.
    
    Parameters
    ----------
    low_color : tuple
        RGB tuple for the low end of the colormap (default is green).
    mid_color : tuple
        RGB tuple for the middle of the colormap (default is white).
    high_color : tuple
        RGB tuple for the high end of the colormap (default is red).

    Returns
    -------
    function
        A function that takes a value, min, max, and optional mid, and returns a reportlab color.
    """

    def _map(val, vmin, vmax, vmid=None):
        if vmax == vmin:  # avoid division by zero
            return colors.Color(*mid_color if mid_color else low_color)

        # default mid = midpoint
        if vmid is None:
            vmid = (vmax + vmin) / 2.0

        if val <= vmid:
            # interpolate low → mid
            t = (val - vmin) / (vmid - vmin) if vmid > vmin else 0
            r = low_color[0] + t * (mid_color[0] - low_color[0])
            g = low_color[1] + t * (mid_color[1] - low_color[1])
            b = low_color[2] + t * (mid_color[2] - low_color[2])
        else:
            # interpolate mid → high
            t = (val - vmid) / (vmax - vmid) if vmax > vmid else 0
            r = mid_color[0] + t * (high_color[0] - mid_color[0])
            g = mid_color[1] + t * (high_color[1] - mid_color[1])
            b = mid_color[2] + t * (high_color[2] - mid_color[2])

        return colors.Color(r, g, b)

    return _map

@dataclass
class PdfColumnFormat( BaseColumnFormat ):
    formatter: Optional[Callable[[Any], str]] = None  # e.g., lambda x: f"${x:,.2f}"
    colormap: Optional[Callable[[float, float, float], colors.Color]] = None
    # colormap takes (val, vmin, vmax) and returns a ReportLab Color


@dataclass
class PdfColumnMeta(  BaseColumnMeta ):
    pass

def build_table_from_df(df: pd.DataFrame, schema: list[PdfColumnMeta]) -> Table:
    """Turn DataFrame + schema into a styled ReportLab Table.
    
    Parameters
    ----------
    df : pd.DataFrame
        The data to display.
    schema : list of PdfColumnMeta
        Metadata for each column, including formatting and colormap info.

    Returns
    -------
    Table
        A ReportLab Table object ready for inclusion in a PDF.
    """
    # --- Build data matrix (header + rows)
    # Take the label if exists, else name as the header of the column
    headers = [col.display_name or col.name for col in schema]
    table_data = [headers]

    # Precompute vmin/vmax for each col needing a colormap
    vmin_vmax = {}
    for col in schema:
        if col.format.colormap:
            series = pd.to_numeric(df[col.name], errors="coerce")
            vmin_vmax[col.name] = (series.min(), series.max())

    # Process each row and each cell with appropriate formatting
    for _, row in df.iterrows():
        row_data = []
        for col in schema:
            val = row[col.name]

            if col.format.transformer:
                val = col.format.transformer(val)
            if col.format.formatter:
                display_val = col.format.formatter(val)
            elif isinstance(val, (int, float, np.number)):
                display_val = default_number_formatter(
                    val, decimals=col.format.decimals, comma=col.format.comma
                )
            else:
                display_val = str(val)
            row_data.append(display_val)
        table_data.append(row_data)

    # --- Build ReportLab table
    tbl = Table(table_data, repeatRows=1)
    style = TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),  # header
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ])

    # --- Apply colormap cell backgrounds
    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        for col_idx, col in enumerate(schema):
            if col.format.colormap:
                vmin, vmax = vmin_vmax[col.name]
                val = row[col.name]
                if pd.notna(val):
                    bgcolor = col.format.colormap(val, vmin, vmax)
                    style.add("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), bgcolor)

    tbl.setStyle(style)
    return tbl
