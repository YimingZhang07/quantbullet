import numpy as np
import matplotlib.pyplot as plt

def get_grid_fig_axes( n_charts, n_cols=3, width=5, height=4 ):
    """Get the fig and flattened axes from a grid"""
    n_rows = int( np.ceil( n_charts / n_cols ) )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=( width * n_cols, height * n_rows))

    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    return fig, axes

def close_unused_axes( axes ):
    """Close unused axes in a grid of axes"""
    for ax in axes:
        has_data = len(ax.lines) > 0 or len(ax.patches) > 0 or len(ax.collections) > 0 or len(ax.images) > 0
        if not has_data:
            ax.set_visible(False)

def scale_scatter_sizes(nums, min_size=30, max_size=300, global_min=None, global_max=None):
    if global_min is None:
        global_min = np.min(nums)
    if global_max is None:
        global_max = np.max(nums)

    # Avoid division by zero
    if global_max == global_min:
        return np.full_like(nums, fill_value=min_size)

    return min_size + (max_size - min_size) * (nums - global_min) / (global_max - global_min)