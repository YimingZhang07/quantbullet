"""
Tests: tests/test_validation
"""

import pandas as pd
import numpy as np
from typing import Sequence

def validate_range_index(x):
    """Validate that the index of a dataframe or series is a RangeIndex from 0."""
    if isinstance(x, pd.DataFrame):
        if not x.index.equals(pd.RangeIndex(start=0, stop=len(x))):
            raise ValueError("Dataframe index is not a RangeIndex from 0.")
    elif isinstance(x, pd.Series):
        if not x.index.equals(pd.RangeIndex(start=0, stop=len(x))):
            raise ValueError("Series index is not a RangeIndex from 0.")
    else:
        pass

def is_same_index_range(x, y):
    if not hasattr(x, 'index') or not hasattr(y, 'index'):
        raise ValueError("x and y must be pandas Series or DataFrame")
    return all(x.index == y.index)

def are_columns_in_df(df, columns):
    """Check if columns are in a dataframe."""
    return all(col in df.columns for col in columns)

def are_only_values_in_series(series, values):
    """Check if a series contains only certain values."""
    return series.isin(values).all()

def is_index_mono_inc(series):
    """Check if the index of a series is monotonically increasing."""
    return series.index.is_monotonic_increasing

def is_array_like(obj):
    return (
        hasattr(obj, '__iter__') and
        not isinstance(obj, (str, bytes, dict)) and
        isinstance(obj, (list, tuple, set, range, pd.Series, np.ndarray))
    )

def is_index_datetime(obj):
    """Check if the index of a Series or DataFrame is of datetime type."""
    return pd.api.types.is_datetime64_any_dtype(obj.index)

class Validator:
    pass

setattr(Validator, 'is_index_mono_inc', staticmethod(is_index_mono_inc))
setattr(Validator, 'is_array_like', staticmethod(is_array_like))
setattr(Validator, 'is_index_datetime', staticmethod(is_index_datetime))