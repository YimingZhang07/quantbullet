"""
Tests:  tests.test_validation
"""

import pandas as pd

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

def are_columns_in_df(df, columns):
    """Check if columns are in a dataframe."""
    return all(col in df.columns for col in columns)

def are_only_values_in_series(series, values):
    """Check if a series contains only certain values."""
    return series.isin(values).all()
