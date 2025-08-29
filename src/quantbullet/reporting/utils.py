import matplotlib.pyplot as plt
import importlib.resources
import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PyPDF2 import PdfMerger

def copy_axis(src_ax: plt.Axes, dst_ax: plt.Axes,
              with_legend: bool = True, 
              with_title: bool = True, 
              with_grid: bool = True, 
              with_ticks: bool = True):
    """Safely replicate the contents of one Axes into another."""
    # Copy lines
    for line in src_ax.get_lines():
        dst_ax.plot(
            line.get_xdata(), line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            label=line.get_label() if line.get_label() != "_nolegend_" else None,
        )

    # Scatter & other collections (like PathCollections from scatter)
    for col in src_ax.collections:
        try:
            offsets = col.get_offsets()
            sizes = col.get_sizes()
            facecolors = col.get_facecolors()
            edgecolors = col.get_edgecolors()
            dst_ax.scatter(
                offsets[:, 0], offsets[:, 1],
                s=sizes, facecolors=facecolors, edgecolors=edgecolors,
                marker=col.get_paths()[0] if col.get_paths() else "o"
            )
        except Exception:
            # Fallback: skip complex collection types
            pass

    # Images (e.g. heatmaps)
    for im in src_ax.images:
        dst_ax.imshow(
            im.get_array(),
            cmap=im.get_cmap(),
            norm=im.norm,
            extent=im.get_extent(),
            origin=im.origin,
            aspect=im.get_aspect(),
            interpolation=im.get_interpolation()
        )

    # Axis limits and scales
    dst_ax.set_xlim(src_ax.get_xlim())
    dst_ax.set_ylim(src_ax.get_ylim())
    dst_ax.set_xscale(src_ax.get_xscale())
    dst_ax.set_yscale(src_ax.get_yscale())

    # Labels
    dst_ax.set_xlabel(src_ax.get_xlabel())
    dst_ax.set_ylabel(src_ax.get_ylabel())

    # Title
    if with_title and src_ax.get_title():
        dst_ax.set_title(src_ax.get_title())

    # Legend
    if with_legend and src_ax.get_legend() is not None:
        handles, labels = src_ax.get_legend_handles_labels()
        dst_ax.legend(handles, labels)

    # --- Ticks ---
    if with_ticks:
        # formatters
        dst_ax.xaxis.set_major_formatter(src_ax.xaxis.get_major_formatter())
        dst_ax.yaxis.set_major_formatter(src_ax.yaxis.get_major_formatter())
        dst_ax.xaxis.set_minor_formatter(src_ax.xaxis.get_minor_formatter())
        dst_ax.yaxis.set_minor_formatter(src_ax.yaxis.get_minor_formatter())

        # locators
        dst_ax.xaxis.set_major_locator(src_ax.xaxis.get_major_locator())
        dst_ax.yaxis.set_major_locator(src_ax.yaxis.get_major_locator())
        dst_ax.xaxis.set_minor_locator(src_ax.xaxis.get_minor_locator())
        dst_ax.yaxis.set_minor_locator(src_ax.yaxis.get_minor_locator())

        # Formatters & locators
        dst_ax.xaxis.set_major_formatter(src_ax.xaxis.get_major_formatter())
        dst_ax.yaxis.set_major_formatter(src_ax.yaxis.get_major_formatter())
        dst_ax.xaxis.set_minor_formatter(src_ax.xaxis.get_minor_formatter())
        dst_ax.yaxis.set_minor_formatter(src_ax.yaxis.get_minor_formatter())

        dst_ax.xaxis.set_major_locator(src_ax.xaxis.get_major_locator())
        dst_ax.yaxis.set_major_locator(src_ax.yaxis.get_major_locator())
        dst_ax.xaxis.set_minor_locator(src_ax.xaxis.get_minor_locator())
        dst_ax.yaxis.set_minor_locator(src_ax.yaxis.get_minor_locator())

    # --- Grid ---
    if with_grid:
        # Major gridlines = just check if any visible gridline exists
        major_on = any(gl.get_visible() for gl in src_ax.get_xgridlines() + src_ax.get_ygridlines())
        dst_ax.grid(major_on, which="major")

        # Minor gridlines = filter by line properties (if available)
        all_x = src_ax.get_xgridlines()
        all_y = src_ax.get_ygridlines()
        # Matplotlib marks minor gridlines with `_is_minor` internally
        minor_lines = [gl for gl in all_x + all_y if getattr(gl, "_is_minor", False)]
        minor_on = any(gl.get_visible() for gl in minor_lines)
        dst_ax.grid(minor_on, which="minor")

def copy_figure(src_fig: plt.Figure, dst_fig: plt.Figure, include_margins: bool = False, include_spacing: bool = True):
    """
    Copy subplot spacing/layout params from src_fig to dst_fig.
    """
    sp = src_fig.subplotpars

    if include_margins:
        dst_fig.subplots_adjust(
            left=sp.left,
            right=sp.right,
            bottom=sp.bottom,
            top=sp.top,
        )

    if include_spacing:
        dst_fig.subplots_adjust(
            wspace=sp.wspace,
            hspace=sp.hspace,
        )

def register_fonts_from_package():
    # fonts/ lives inside your_package
    font_dir = importlib.resources.files("quantbullet.reporting.fonts")

    for font_file in font_dir.iterdir():
        if font_file.suffix.lower() == ".ttf":
            # Use filename (without extension) as fontName
            font_name = font_file.stem
            pdfmetrics.registerFont(TTFont(font_name, str(font_file)))


def merge_pdfs(pdf_paths, output_path):
    """Merge multiple PDFs into a single PDF."""
    merger = PdfMerger()
    for pdf in pdf_paths:
        merger.append(pdf)

    # if the write is failed, need to close the pdfs
    try:
        merger.write(output_path)
    except:
        merger.close()
        raise
    merger.close()
    return output_path