import numpy as np
import pandas as pd
from typing import Union
from ..core.type_hints import ArrayLike

def fit_constant_shift(
    x: ArrayLike,
    y: ArrayLike,
) -> float:
    """Find the constant shift between two series."""
    # Convert to np.ndarray
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Find common non-NaN entries
    mask = ~np.isnan(x_arr) & ~np.isnan(y_arr)
    x_obs = x_arr[mask]
    y_obs = y_arr[mask]

    if len(x_obs) == 0:
        raise ValueError("No overlapping non-NaN data between x and y.")

    # Closed-form solution
    delta = y_obs.mean() - x_obs.mean()
    return delta

def align_series_index(
    x1: pd.Series,
    x2: pd.Series,
):
    """Align the index of two series so that they have the same index."""
    # Find the common index
    common_index = x1.index.intersection(x2.index)

    # Align the series to the common index
    x1_aligned = x1.reindex(common_index)
    x2_aligned = x2.reindex(common_index)

    return x1_aligned, x2_aligned