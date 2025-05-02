import pandas as pd
import numpy as np
from typing import Sequence
from datetime import datetime, date

class Consolidator:
    @staticmethod
    def consolidate_to_series(data: Sequence) -> pd.Series:
        """Convert sequence-like data to pandas Series, validating shape and index."""
        if isinstance(data, pd.Series):
            if not data.index.is_monotonic_increasing:
                raise ValueError("Input Series index must be monotonic increasing.")
            return data
        elif isinstance(data, (list, tuple, np.ndarray)):
            return pd.Series(data)
        else:
            raise TypeError("Data must be a Series, list, tuple, or ndarray.")
        
    @staticmethod
    def to_time_stamp( dt ) -> pd.Timestamp:
        """Convert various date types to pandas Timestamp."""
        if isinstance(dt, pd.Timestamp):
            return dt.normalize()
        elif isinstance(dt, (datetime, date)):
            return pd.Timestamp(dt).normalize()
        elif isinstance(dt, np.datetime64):
            return pd.Timestamp(dt).normalize()
        elif isinstance(dt, str):
            return pd.to_datetime(dt).normalize()
        else:
            raise TypeError(f"Unsupported date type: {type(dt)}")