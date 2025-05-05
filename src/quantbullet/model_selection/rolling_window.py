import pandas as pd
from quantbullet.utils.consolidator import Consolidator
from ..model.core import Model
from concurrent.futures import ThreadPoolExecutor

class RollingWindowManager:
    """
    A class to manage rolling windows of data for time series analysis.
    
    It allows for the retrieval of data within a specified window size and end date.
    The data is expected to be in a DataFrame with a multi-level index, where one of the levels is a date.
    
    The class also provides functionality to retrieve the next day's data based on the current date.
    """
    def __init__(self, df: pd.DataFrame, date_level_name: str = 'date', window_size: int = None):
        if date_level_name not in df.index.names:
            raise ValueError(f"'{date_level_name}' must be in the index names of the DataFrame.")
        
        self._date_level = date_level_name
        self.df = df.sort_index()
        self.all_dates = self.df.index.get_level_values(self._date_level).unique().sort_values()
        self.return_flat = False
        
        # provide instance level control over window size and full history
        self._window_size = window_size
        self._use_full_history = False
        
    @property
    def window_size(self) -> int:
        """Get the current window size."""
        return self._window_size
        
    @window_size.setter
    def window_size(self, value: int):
        if value <= 0:
            raise ValueError("Window size must be a positive integer.")
        self._window_size = value
        
    @property
    def use_full_history(self) -> bool:
        """Get the current setting for using full history."""
        return self._use_full_history
    
    @use_full_history.setter
    def use_full_history(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("use_full_history must be a boolean value.")
        self._use_full_history = value
        
    def get_first_window_date(self) -> pd.Timestamp:
        """Get the first date that is available for the rolling window."""
        if self.window_size is None:
            raise ValueError("Window size is not set.")
        if len(self.all_dates) < self.window_size:
            raise ValueError("Not enough data for the requested window size.")
        return self.all_dates[self.window_size - 1]
    
    def get_next_date(self, current_date) -> pd.Timestamp:
        """Get the next date after the current date."""
        current_date = Consolidator.to_time_stamp(current_date)
        idx = self.all_dates.get_loc(current_date)
        if idx + 1 >= len(self.all_dates):
            raise ValueError("No more rolling dates available.")
        return self.all_dates[idx + 1]
    
    def get_all_window_dates(self) -> pd.DataFrame:
        """Get all dates available for the rolling window."""
        first_date = self.get_first_window_date()
        last_date = self.all_dates[-2]
        window_dates = self.all_dates[(self.all_dates >= first_date) & (self.all_dates <= last_date)]
        return window_dates

    def get_window_data(self, end_date, window_size: int = None, use_full_history: bool = False) -> pd.DataFrame:
        """
        Get a rolling window of data ending at the specified date.
        
        Parameters
        ----------
        end_date : str or pd.Timestamp
            The end date for the rolling window.
        window_size : int, optional
            The size of the window. If None, the full history up to end_date is used.
        use_full_history : bool, optional
            If True, the full history up to end_date is used. If False, the window size is used.
        """
        end_date = Consolidator.to_time_stamp(end_date)
        if end_date not in self.all_dates:
            raise ValueError(f"End date {end_date} not in available dates.")
        
        if window_size is None and self.window_size is not None:
            window_size = self.window_size
        
        if window_size is None and self._window_size is None:
            raise ValueError("You must specify a window_size or set it in the instance.")

        end_idx = self.all_dates.get_loc(end_date)

        if use_full_history or self._use_full_history:
            window_dates = self.all_dates[: end_idx + 1]
        else:
            if end_idx + 1 < window_size:
                raise ValueError("Not enough data for the requested window size.")
            window_dates = self.all_dates[end_idx + 1 - window_size : end_idx + 1]

        result = self.df.loc[self.df.index.get_level_values(self._date_level).isin(window_dates)]
        return self._format_return(result)


    def get_next_day_data(self, current_date) -> pd.DataFrame | None:
        try:
            next_day = self.get_next_date(current_date)
        except ValueError as e:
            raise ValueError("No next day data available.") from e
        
        result = self.df.loc[self.df.index.get_level_values(self._date_level) == next_day]
        return self._format_return(result)
    
    def _format_return(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.reset_index() if self.return_flat else df
    
    @classmethod
    def from_flat( cls, df: pd.DataFrame, date_col: str = 'date', window_size: int = None):
        """
        Create an instance of RollingWindowManager from a flat DataFrame.
        
        A flat DataFrame means that it does not have an index set, and the date column is a regular column.
        
        The DataFrame must contain a date column that will be used as the index.
        
        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be converted.
        date_col : str
            The name of the date column in the DataFrame.
            
        Returns
        -------
        RollingWindowManager
            An instance of RollingWindowManager with the date column set as the index.
        """
        if date_col not in df.columns:
            raise ValueError(f"'{date_col}' must be a column in the DataFrame.")
        
        df_sorted = df.sort_values(by=date_col)
        df_indexed = df_sorted.set_index(date_col)
        
        instance = cls(df_indexed, date_level_name=date_col, window_size=window_size)
        instance.return_flat = True
        return instance


class RollingWindowResult:
    def __init__(self, run_date, data, result, model):
        self.run_date = run_date
        self.data = data
        self.result = result
        self.model = model

    def __repr__(self):
        return f"RollingWindowResult(run_date={self.run_date}"

class RollingWindowResultLogger:
    def __init__( self, target_col: str ):
        self._results = dict()
        self._target_col = target_col

    def log( self, run_date, data, model, preds ):
        """log the results of the rolling window backtest"""
        # assume the data is a DataFrame, we will append the result to the data
        log_data = data.copy()
        log_data[ 'loggerPred' ]      = preds
        log_data[ 'loggerRunDate' ]   = run_date
        log_data[ 'loggerResid' ]     = log_data[ self._target_col ] - log_data[ 'loggerPred' ]
        self._results[run_date]       = RollingWindowResult( run_date, log_data, preds, model )

    def to_dataframe( self ) -> pd.DataFrame:
        """convert the results to a DataFrame"""
        if len(self._results) == 0:
            return pd.DataFrame()
        result_df = pd.concat([res.data for res in self._results.values()])
        return result_df.reset_index(drop=True)

class RollingWindowBacktester:
    def __init__( self, 
                    rolling_window_manager: RollingWindowManager,
                    model: Model ):
        self.manager = rolling_window_manager
        self.model = model
        # the logger has to at least know which column is the target to calculate the residuals
        self.loggger = RollingWindowResultLogger( self.model.feature_spec.y )

    def _prepare( self ):
        self._back_test_dates = self.manager.get_all_window_dates()
        # make sure the rollling window mangager has all the data needed for the model
        required_features = self.model.feature_spec.all_features
        missing_features = set(required_features) - set(self.manager.df.columns)
        if len(missing_features) > 0:
            raise ValueError(f"Missing features in the RollingWindowManager df: {missing_features}")

    def run( self, use_multithreading: bool = False ):
        self._prepare()
        if use_multithreading:
            with ThreadPoolExecutor() as executor:
                executor.map(self._run, self._back_test_dates)
        else:
            for run_date in self._back_test_dates:
                self._run( run_date )

    def _run( self, run_date ):
        window_data = self.manager.get_window_data( run_date )
        next_day_data = self.manager.get_next_day_data( run_date )

        # make sure the model is a new instance for each run
        # this is important to avoid data leakage between runs especially as we are using multithreading
        model = self.model.new_instance()

        model.fit( window_data )
        preds = model.predict( next_day_data )
        self.loggger.log( run_date, next_day_data, model, preds )

    def get_logger_df( self ):
        return self.loggger.to_dataframe()