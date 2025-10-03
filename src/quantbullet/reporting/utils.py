import matplotlib.pyplot as plt
import importlib.resources
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfWriter

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


    # Copy patches (bars, rectangles, etc.)
    for patch in src_ax.patches:
        if hasattr(patch, 'get_xy') and hasattr(patch, 'get_width') and hasattr(patch, 'get_height'):
            # Handle Rectangle patches (bar charts)
            x, y = patch.get_xy()
            width = patch.get_width()
            height = patch.get_height()
            
            dst_ax.bar(
                x + width/2,  # bar() expects center position
                height,
                width=width,
                bottom=y,
                color=patch.get_facecolor(),
                edgecolor=patch.get_edgecolor(),
                linewidth=patch.get_linewidth(),
                alpha=patch.get_alpha(),
                label=patch.get_label() if hasattr(patch, 'get_label') and patch.get_label() != "_nolegend_" else None
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

    # Enhanced Ticks with Rotation Support
    if with_ticks:
        # Copy formatters and locators
        dst_ax.xaxis.set_major_formatter(src_ax.xaxis.get_major_formatter())
        dst_ax.yaxis.set_major_formatter(src_ax.yaxis.get_major_formatter())
        dst_ax.xaxis.set_minor_formatter(src_ax.xaxis.get_minor_formatter())
        dst_ax.yaxis.set_minor_formatter(src_ax.yaxis.get_minor_formatter())
        dst_ax.xaxis.set_major_locator(src_ax.xaxis.get_major_locator())
        dst_ax.yaxis.set_major_locator(src_ax.yaxis.get_major_locator())
        dst_ax.xaxis.set_minor_locator(src_ax.xaxis.get_minor_locator())
        dst_ax.yaxis.set_minor_locator(src_ax.yaxis.get_minor_locator())
        
        # Copy tick label properties (including rotations)
        # X-axis major ticks
        src_x_major = src_ax.get_xticklabels()
        dst_x_major = dst_ax.get_xticklabels()
        for src_tick, dst_tick in zip(src_x_major, dst_x_major):
            dst_tick.set_rotation(src_tick.get_rotation())
            dst_tick.set_horizontalalignment(src_tick.get_horizontalalignment())
            dst_tick.set_verticalalignment(src_tick.get_verticalalignment())
            dst_tick.set_fontsize(src_tick.get_fontsize())
            dst_tick.set_color(src_tick.get_color())
            dst_tick.set_weight(src_tick.get_weight())
            dst_tick.set_style(src_tick.get_style())
        
        # Y-axis major ticks
        src_y_major = src_ax.get_yticklabels()
        dst_y_major = dst_ax.get_yticklabels()
        for src_tick, dst_tick in zip(src_y_major, dst_y_major):
            dst_tick.set_rotation(src_tick.get_rotation())
            dst_tick.set_horizontalalignment(src_tick.get_horizontalalignment())
            dst_tick.set_verticalalignment(src_tick.get_verticalalignment())
            dst_tick.set_fontsize(src_tick.get_fontsize())
            dst_tick.set_color(src_tick.get_color())
            dst_tick.set_weight(src_tick.get_weight())
            dst_tick.set_style(src_tick.get_style())
        
        # Copy minor tick labels if they exist
        try:
            src_x_minor = src_ax.get_xticklabels(minor=True)
            dst_x_minor = dst_ax.get_xticklabels(minor=True)
            for src_tick, dst_tick in zip(src_x_minor, dst_x_minor):
                dst_tick.set_rotation(src_tick.get_rotation())
                dst_tick.set_horizontalalignment(src_tick.get_horizontalalignment())
                dst_tick.set_verticalalignment(src_tick.get_verticalalignment())
                dst_tick.set_fontsize(src_tick.get_fontsize())
                dst_tick.set_color(src_tick.get_color())
        except:
            pass
            
        try:
            src_y_minor = src_ax.get_yticklabels(minor=True)
            dst_y_minor = dst_ax.get_yticklabels(minor=True)
            for src_tick, dst_tick in zip(src_y_minor, dst_y_minor):
                dst_tick.set_rotation(src_tick.get_rotation())
                dst_tick.set_horizontalalignment(src_tick.get_horizontalalignment())
                dst_tick.set_verticalalignment(src_tick.get_verticalalignment())
                dst_tick.set_fontsize(src_tick.get_fontsize())
                dst_tick.set_color(src_tick.get_color())
        except:
            pass

    # --- Grid ---
    if with_grid:
        # Major on/off detection from existing grid lines
        major_on = any(gl.get_visible() and not getattr(gl, "_is_minor", False)
                       for gl in src_ax.get_xgridlines() + src_ax.get_ygridlines())
        dst_ax.grid(major_on, which="major")

        # Minor on/off detection
        minor_on = any(gl.get_visible() and getattr(gl, "_is_minor", False)
                       for gl in src_ax.get_xgridlines() + src_ax.get_ygridlines())
        dst_ax.grid(minor_on, which="minor")

        # Copy all gridline styles
        for src_gl, dst_gl in zip(src_ax.get_xgridlines() + src_ax.get_ygridlines(),
                                 dst_ax.get_xgridlines() + dst_ax.get_ygridlines()):
            dst_gl.set_linestyle(src_gl.get_linestyle())
            dst_gl.set_linewidth(src_gl.get_linewidth())
            dst_gl.set_color(src_gl.get_color())
            dst_gl.set_alpha(src_gl.get_alpha())

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
    merger = PdfWriter()
    for pdf in pdf_paths:
        merger.append(pdf)
    try:
        with open(output_path, "wb") as f_out:
            merger.write(f_out)
    finally:
        merger.close()
    return output_path