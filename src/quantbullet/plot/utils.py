import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

def get_grid_fig_axes( n_charts, n_cols=3, width=5, height=4, flatten=True ):
    """Get the fig and axes from a grid
    
    Parameters
    ----------
    n_charts : int
        Number of charts to plot
    n_cols : int, optional
        Number of columns in the grid, by default 3
    width : int, optional
        Width of each subplot, by default 5
    height : int, optional
        Height of each subplot, by default 4
    flatten : bool, optional
        Whether to flatten the axes array, by default True

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : np.ndarray
        The axes array
    """
    n_rows = int( np.ceil( n_charts / n_cols ) )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=( width * n_cols, height * n_rows))

    if n_rows * n_cols == 1:
        axes = np.array([axes])

    if flatten:
        axes = axes.flatten()
    return fig, axes

def close_unused_axes( axes ):
    """Close unused axes in a grid of axes"""
    for ax in axes:
        has_data = len(ax.lines) > 0 or len(ax.patches) > 0 or len(ax.collections) > 0 or len(ax.images) > 0
        if not has_data:
            ax.set_visible(False)

def scale_scatter_sizes(nums, min_size=20, max_size=100, global_min=None, global_max=None, power: float = 1.0):
    """
    Scale numeric values into scatter marker areas.

    Parameters
    ----------
    nums : array-like
        Values to scale (in original units, e.g. counts).
    power : float
        Apply a power transform before linear scaling into [min_size, max_size].
        - power=1.0: linear (default, backward compatible)
        - power=0.5: sqrt compression (common for bubble charts)
    """
    if power <= 0:
        raise ValueError("power must be > 0")

    nums_arr = np.asarray(nums, dtype=float)
    nums_scaled = nums_arr ** float(power)

    if global_min is None:
        global_min = float(np.min(nums_arr))
    if global_max is None:
        global_max = float(np.max(nums_arr))

    global_min_scaled = float(global_min) ** float(power)
    global_max_scaled = float(global_max) ** float(power)

    # Avoid division by zero
    if global_max_scaled == global_min_scaled:
        return np.full_like(nums_arr, fill_value=min_size)

    return min_size + (max_size - min_size) * (nums_scaled - global_min_scaled) / (global_max_scaled - global_min_scaled)


def pretty_int_breaks(vmin, vmax, n: int = 3) -> list[int]:
    """
    Generate "pretty" integer break values between vmin and vmax (ggplot-like).

    Uses Matplotlib's MaxNLocator (1/2/5×10^k stepping) and then filters/dedupes.
    Intended for legend reference values, not strict statistical summaries.
    """
    if vmin is None or vmax is None:
        return []
    vmin = float(vmin)
    vmax = float(vmax)
    if np.isnan(vmin) or np.isnan(vmax):
        return []
    if vmin > vmax:
        vmin, vmax = vmax, vmin
    if vmin == vmax:
        return [int(round(vmin))]

    n = max(int(n), 2)
    locator = mticker.MaxNLocator(nbins=n, integer=True, prune=None)
    ticks = locator.tick_values(vmin, vmax)
    ticks = [int(round(t)) for t in ticks if vmin <= t <= vmax]

    # Deduplicate, preserve order
    out = []
    for t in ticks:
        if t not in out:
            out.append(t)

    # If locator returns too many, pick ~evenly spaced values
    if len(out) > n:
        idx = np.linspace(0, len(out) - 1, n).round().astype(int)
        out = [out[i] for i in idx]
        out = sorted(set(out))

    # If it returns too few (tight range), fall back to endpoints
    if len(out) < 2:
        out = sorted(set([int(round(vmin)), int(round(vmax))]))

    return out