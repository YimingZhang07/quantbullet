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