import copy
import matplotlib.pyplot as plt
import numpy as np

def copy_axis(src_ax: plt.Axes, dst_ax: plt.Axes,
              with_legend: bool = True, with_title: bool = True):
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