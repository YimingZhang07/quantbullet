"""
Tests: tests/test_validation
"""

import pandas as pd
import numpy as np

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

   
class Consolidator:
    @staticmethod
    def consolidate_to_series(data):
        if not isinstance(data, (pd.Series, list, tuple, np.ndarray)):
            raise ValueError("Input data must be a pandas Series, list, tuple, or numpy array.")

        if isinstance(data, pd.Series):
            if not data.index.is_monotonic:
                raise ValueError("Input pandas Series must have a monotonic index.")
            return data

        try:
            return pd.Series(data)
        except TypeError:
            raise TypeError("Input data cannot be converted to a pandas Series.")

class Validator:
    pass

setattr(Validator, 'is_index_mono_inc', staticmethod(is_index_mono_inc))