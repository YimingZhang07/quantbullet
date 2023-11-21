import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator

class TimeSeriesDailyRollingSplit:
    def __init__(self,
                 min_train_size: int=None,
                 max_train_size: int=None) -> None:
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
    
    def split(self,
              df: pd.DataFrame,
              date_column: str = 'date') -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Split a dataframe into train and test sets based on a date column, which is rolled forward daily.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to split.
        date_column : str, optional
            Column name of the date column, by default 'date'.

        Returns
        -------
        Iterator of tuples of train and test indices.
            Iterator of tuples of train and test sets.
        """
        df = df.sort_values(by=date_column)
        df = df.reset_index(drop=True)
        df[date_column] = pd.to_datetime(df[date_column])
        min_train_size = self.min_train_size if self.min_train_size else 0
        max_train_size = self.max_train_size if self.max_train_size else len(df)
        unique_dates = df[date_column].unique()
        for end_date in unique_dates:
            train_mask = df[date_column] < end_date
            # the scikit-learn convention is to return indices and must be a numpy array
            train_indices = np.array(df[train_mask].index)
            if len(train_indices) <= min_train_size or len(train_indices) >= max_train_size:
                continue
            test_mask = df[date_column] == end_date
            test_indices = np.array(df[test_mask].index)
            yield (train_indices, test_indices)
