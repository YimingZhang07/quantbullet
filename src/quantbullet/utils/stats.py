import pandas as pd
import numpy as np
from .validation import is_same_index_range

def _convert_sequence_to_series(x, y):
    if type(x) != type(y):
        raise ValueError("x and y must be of the same type")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if isinstance(x, np.ndarray) or isinstance(x, list):
        return pd.Series(x), pd.Series(y)
    if isinstance(x, pd.Series):
        if is_same_index_range(x, y):
            return x, y
        else:
            return pd.Series(x.values), pd.Series(y.values)

def cross_correlation(x, y, k):
    """
    Compute the cross-correlation between two sequences x and y for lags from -k to k.

    Notes
    -----
    The cross-correlation holds x constant and shifts y by the lag. When the lag is negative, y is in the future.

    Parameters
    ----------
    x : array-like
        The first sequence.
    y : array-like
        The second sequence.
    k : int
        The maximum lag.
    
    Returns
    -------
    cross_correlation : pd.Series
        The cross-correlation for lags from -k to k.
    """
    x, y = _convert_sequence_to_series(x, y)
    res = pd.Series(dtype=float, index=range(-k, k+1), name='cross_correlation')
    for order in range(-k, k+1):
        y_ = y.shift(order).dropna()
        x_ = x.loc[y_.index]
        res.loc[order] = np.corrcoef(x_, y_)[0][1]
    return res