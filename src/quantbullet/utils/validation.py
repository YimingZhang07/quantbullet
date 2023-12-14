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
