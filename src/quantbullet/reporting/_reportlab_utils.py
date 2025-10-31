from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import math
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

from .base import BaseColumnFormat, BaseColumnMeta
from .formatters import flex_number_formatter

def to_reportlab_color(value):
    """Convert a hex string or RGB tuple into a reportlab Color object."""
    if isinstance(value, colors.Color):
        return value  # already a Color object

    if isinstance(value, str):
        # handle hex color like '#RRGGBB' or '#RGB'
        value = value.strip()
        if value.startswith("#"):
            hex_value = value.lstrip("#")
            if len(hex_value) == 3:
                hex_value = "".join(c * 2 for c in hex_value)
            r, g, b = [int(hex_value[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
            return colors.Color(r, g, b)
        else:
            # also allow named colors like "red", "blue", etc.
            return getattr(colors, value.lower(), colors.Color(0, 0, 0))

    elif isinstance(value, (tuple, list)) and len(value) == 3:
        # assume tuple/list of RGB in 0–1 or 0–255 scale
        if max(value) > 1:  # detect 0–255 scale
            r, g, b = [v / 255.0 for v in value]
        else:
            r, g, b = value
        return colors.Color(r, g, b)

    else:
        raise ValueError(f"Unsupported color format: {value!r}")

# This function should have been deprecated in favor of to_reportlab_color
# def hex_to_rgb01(hex_str: str):
#     """Convert a hex color string to an RGB tuple with values between 0 and 1."""
#     hex_str = hex_str.lstrip("#")
#     return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def make_diverging_colormap(low_color=(0.8, 0, 0), mid_color=(1, 1, 1), high_color=(0, 0.8, 0), vmid=None):
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
    low_color = to_reportlab_color(low_color)
    mid_color = to_reportlab_color(mid_color)
    high_color = to_reportlab_color(high_color)

    def _map(val, vmin, vmax, vmid_override=None):
        _vmid = vmid_override if vmid_override is not None else vmid
        if vmax == vmin:  # avoid division by zero
            return colors.Color(*mid_color if mid_color else low_color)

        # default mid = midpoint
        if _vmid is None:
            _vmid = (vmax + vmin) / 2.0

        if val <= _vmid:
            # interpolate low → mid
            t = (val - vmin) / (_vmid - vmin) if _vmid > vmin else 0
            r = low_color.red + t * (mid_color.red - low_color.red)
            g = low_color.green + t * (mid_color.green - low_color.green)
            b = low_color.blue + t * (mid_color.blue - low_color.blue)
        else:
            # interpolate mid → high
            t = (val - _vmid) / (vmax - _vmid) if vmax > _vmid else 0
            r = mid_color.red + t * (high_color.red - mid_color.red)
            g = mid_color.green + t * (high_color.green - mid_color.green)
            b = mid_color.blue + t * (high_color.blue - mid_color.blue)

        return colors.Color(r, g, b)

    return _map

@dataclass
class PdfColumnFormat( BaseColumnFormat ):
    formatter: Optional[Callable[[Any], str]] = None  # e.g., lambda x: f"${x:,.2f}"
    colormap: Optional[Callable[[float, float, float], colors.Color]] = None  # colormap takes (val, vmin, vmax) and returns a ReportLab Color
    named_colormap: Optional[str] = None # we use a few predefined colormaps by name

    def __post_init__(self):
        if self.named_colormap == "red_white_green":
            self.colormap = make_diverging_colormap( high_color="#63be7b", mid_color=(1,1,1), low_color="#f8696b" )
        elif self.named_colormap == "white_green":
            self.colormap = make_diverging_colormap( high_color="#63be7b", mid_color=(1,1,1), low_color=(1,1,1) )

@dataclass
class PdfColumnMeta(  BaseColumnMeta ):
    # override the format type
    format: PdfColumnFormat = field( default_factory=PdfColumnFormat )

def safe_color(c, default=colors.white):
    """Ensure ReportLab color is finite and valid."""
    if c is None:
        return default
    
    try:
        r, g, b = c.red, c.green, c.blue
        if all(math.isfinite(v) and 0 <= v <= 1 for v in (r, g, b)):
            return c
    except AttributeError:
        pass
    
    return default

def apply_heatmap(table_data, row_range, col_range, cmap, vmid=None):
    """
    Apply heatmap coloring to a region of table_data.

    Parameters
    ----------
    table_data : list of list
        The full table including headers.
    row_range : tuple
        (row_start, row_end) inclusive, in table_data coordinates.
    col_range : tuple
        (col_start, col_end) inclusive.
    cmap : function
        A colormap function from make_diverging_colormap.
    vmid : float, optional
        Mid value for diverging colormap.

    Returns
    -------
    styles : list of tuple
        ReportLab TableStyle commands.
    """
    r0, r1 = row_range
    c0, c1 = col_range

    values = []
    for r in range(r0, r1+1):
        for c in range(c0, c1+1):
            try:
                x = table_data[r][c]
                x = x.replace(",", "").replace("$", "").replace("%", "")
                v = float(x)
                if math.isfinite(v):
                    values.append(v)
            except ( ValueError, TypeError ):
                continue

    if not values:
        return []

    vmin, vmax = min(values), max(values)

    if vmin == vmax:
        vmax = vmin + 1e-5

    if vmid is None:
        vmid = (vmax + vmin) / 2.0

    styles = []

    # we certainly hope the logic should be very robust
    # try not to crash the code when data is bad and colors are not computable
    # the code here is obviously not super efficient and we have done too many repetitive protections

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            try:
                x = str(table_data[r][c]).replace(",", "").replace("$", "").replace("%", "")
                val = float(x)
                if not math.isfinite(val):
                    continue
                color = safe_color(cmap(val, vmin, vmax, vmid))
                styles.append(("BACKGROUND", (c, r), (c, r), color))
            except (ValueError, TypeError):
                continue
    return styles

def build_table_from_df( df: pd.DataFrame, schema: list[PdfColumnMeta], col_widths: list[float] = None ) -> Table:
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
                display_val = flex_number_formatter(
                    val, decimals=col.format.decimals, comma=col.format.comma
                )
            elif pd.isna(val):
                display_val = ""
            else:
                display_val = str(val)
            row_data.append(display_val)
        table_data.append(row_data)

    # --- Build ReportLab table
    if col_widths is not None:
        tbl = Table( table_data, repeatRows=1, colWidths=col_widths )
    else:
        tbl = Table( table_data, repeatRows=1 )
        
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

def multi_index_df_to_table_data(df: pd.DataFrame):
    """
    Convert a MultiIndex DataFrame into ReportLab table data + span styles.

    Returns
    -------
    table_data : list of list
        Table data including headers and data rows.
    spans : list of tuple
        [( "SPAN", (col0,row0), (col1,row1) ), ...]
    """
    table_data = []
    spans = []

    nrow_levels = df.index.nlevels
    ncol_levels = df.columns.nlevels

    # Step 1: Build table header (column MultiIndex)
    col_headers = []
    for level in range(ncol_levels):
        row = [""] * nrow_levels  # Leave space for row index levels
        row += list(df.columns.get_level_values(level))
        col_headers.append(row)
    table_data.extend(col_headers)

    # Step 2: Build data rows
    for idx, row_values in zip(df.index, df.values):
        row = []
        if nrow_levels == 1:
            row.append(idx)
        else:
            row.extend(idx)
        row.extend(row_values.tolist())
        table_data.append(row)

    # Step 3: Handle column-wise span (column MultiIndex)
    for level in range(ncol_levels):
        values = df.columns.get_level_values(level)
        start = 0
        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[start]:
                if i - start > 1:
                    spans.append(
                        ("SPAN", (nrow_levels + start, level), (nrow_levels + i - 1, level))
                    )
                start = i

    # Step 4: Handle row-wise span (row MultiIndex)
    for level in range(nrow_levels):
        values = df.index.get_level_values(level)
        start = 0
        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[start]:
                if i - start > 1:
                    spans.append(
                        ("SPAN", (level, ncol_levels + start), (level, ncol_levels + i - 1))
                    )
                start = i

    return table_data, spans
