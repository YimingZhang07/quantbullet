import numpy as np
import pandas as pd
import datetime
from typing import List, Tuple, Iterator

class TimeSeriesDailyRollingSplit:
    def __init__(self,
                 min_train_size: int=None,
                 max_train_size: int=None) -> None:
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
    
    def split(self,
              df: pd.DataFrame,
              reference_column: str = 'date') -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split a dataframe into train and test sets based on a date column, which is rolled forward daily.
        
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
        df = df.sort_values(by=reference_column)
        df = df.reset_index(drop=True)
        df[reference_column] = pd.to_datetime(df[reference_column])
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
