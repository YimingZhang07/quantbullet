import pandas as pd
from quantbullet.utils.consolidator import Consolidator

class RollingWindowManager:
    """
    A class to manage rolling windows of data for time series analysis.
    
    It allows for the retrieval of data within a specified window size and end date.
    The data is expected to be in a DataFrame with a multi-level index, where one of the levels is a date.
    
    The class also provides functionality to retrieve the next day's data based on the current date.
    """
    def __init__(self, df: pd.DataFrame, date_level_name: str = 'date'):
        if date_level_name not in df.index.names:
            raise ValueError(f"'{date_level_name}' must be in the index names of the DataFrame.")
        
        self._date_level = date_level_name
        self.df = df.sort_index()
        self.all_dates = self.df.index.get_level_values(self._date_level).unique().sort_values()
        self.return_flat = False

    def get_window(self, end_date, window_size: int = None, use_full_history: bool = False) -> pd.DataFrame:
        
        end_date = Consolidator.to_time_stamp(end_date)
        if end_date not in self.all_dates:
            raise ValueError(f"End date {end_date} not in available dates.")

        end_idx = self.all_dates.get_loc(end_date)

        if use_full_history:
            window_dates = self.all_dates[: end_idx + 1]
        else:
            if window_size is None:
                raise ValueError("You must specify a window_size if use_full_history is False.")
            if end_idx + 1 < window_size:
                raise ValueError("Not enough data for the requested window size.")
            window_dates = self.all_dates[end_idx + 1 - window_size : end_idx + 1]

        result = self.df.loc[self.df.index.get_level_values(self._date_level).isin(window_dates)]
        return self._format_return(result)


    def get_next_day_data(self, current_date) -> pd.DataFrame | None:
        current_date = Consolidator.to_time_stamp(current_date)
        idx = self.all_dates.get_loc(current_date)
        if idx + 1 >= len(self.all_dates):
            return None
        next_day = self.all_dates[idx + 1]
        result = self.df.loc[self.df.index.get_level_values(self._date_level) == next_day]
        return self._format_return(result)
    
    def _format_return(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.reset_index() if self.return_flat else df
    
    @classmethod
    def from_flat( cls, df: pd.DataFrame, date_col: str = 'date'):
        if date_col not in df.columns:
            raise ValueError(f"'{date_col}' must be a column in the DataFrame.")
        
        df_sorted = df.sort_values(by=date_col)
        df_indexed = df_sorted.set_index(date_col)
        
        instance = cls(df_indexed, date_level_name=date_col)
        instance.return_flat = True
        return instance