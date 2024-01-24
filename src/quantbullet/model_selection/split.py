import numpy as np
import pandas as pd
import datetime
from typing import List, Tuple, Iterator

class TimeSeriesDailyRollingSplit:
    """a backtesting split that rolls forward daily, compatible with sklearn's cross_val_score."""
    def __init__(self,
                 min_train_size: int=None,
                 max_train_size: int=None) -> None:
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size

    
    def split(self,
              df: pd.DataFrame,
              reference_column: str = 'date') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """generate train and test set indices for a dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to split.
        reference_column : str, optional
            Column name of the date column, by default 'date'.

        Returns
        -------
        List[Tuple[pd.DataFrame, pd.DataFrame]]
            List of tuples of train and test set indices.
        """
        if reference_column not in df.columns:
            raise ValueError(f"Column {reference_column} not in dataframe.")
        if not df[reference_column].is_monotonic_increasing:
            raise ValueError(f"Column {reference_column} is not sorted in ascending order.")
        if not df.index.equals(pd.RangeIndex(start=0, stop=len(df))):
            raise ValueError("Dataframe index is not a RangeIndex from 0. Please ensure that the dataframe is not indexed by date.")
        min_train_size = self.min_train_size if self.min_train_size else 0
        max_train_size = self.max_train_size if self.max_train_size else len(df)
        unique_dates = df[reference_column].unique()
        for end_date in unique_dates:
            train_mask = df[reference_column] < end_date
            train_indices = np.array(df[train_mask].index)
            if len(train_indices) <= min_train_size or len(train_indices) >= max_train_size:
                continue
            test_mask = df[reference_column] == end_date
            test_indices = np.array(df[test_mask].index)
            yield (train_indices, test_indices)
