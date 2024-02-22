import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from .validation import are_only_values_in_series

class BaseDataProvider:
    def __init__(self, data):
        if not isinstance(data, pd.Series) and not isinstance(data, pd.DataFrame):
            raise ValueError('Data must be a pandas Series or DataFrame')
        
        # make a deep copy of the data to avoid modifying the original data when we modify the index type
        self.data = data.copy(deep=True)
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = self.data.index.date
        if not self.data.index.is_monotonic_increasing:
            raise ValueError('Index must be sorted')
        
        self.start = data.index[0]
        self.end = data.index[-1]

class SimpleDataProvider(BaseDataProvider):
    def get_historical_data(self, start_date, end_date):
        # if start_date is not provided, set it to the start of the data
        if start_date is None or start_date == '':
            start_date = self.start
        return self.data.loc[start_date:end_date]

    def get_values_on_date(self, date):
        return self.data.get_loc(date)
        
    
class SimpleSignalProvider(BaseDataProvider):
    def __init__(self, signals):
        if not are_only_values_in_series(signals, [-1, 0, 1, np.nan]):
            raise ValueError('Signals must contain only -1, 0, 1, or np.nan')
        super().__init__(signals)

    def get_signal_on_date(self, date):
        signal = self.data.loc[date]
        if np.isnan(signal):
            return 0
        return signal

class SimplePosition:
    def __init__(self, date, shares):
        self.date = date
        self.shares = shares

    def set_date(self, date):
        self.date = date

    def set_shares(self, shares):
        self.shares = shares

    def __str__(self) -> str:
        return f'SimplePosition({self.date}, {self.shares})'
    
    def __repr__(self) -> str:
        return str(self)

class SimpleBacktest:

    # __slots__ = ['calendar', 'signal_provider', 'closing_signal_provider', 'current_position', 'position_history']

    def __init__(self, calendar, signal_provider, closing_signal_provider):
        self.calendar = calendar
        self.signal_provider = signal_provider
        self.closing_signal_provider = closing_signal_provider
        self.current_position = None

    def initialize(self):
        self.position_history = []

    def arhive_position(self):
        _current_position = copy.deepcopy(self.current_position)
        self.position_history.append(_current_position)

    def run(self):
        self.initialize()
        self.current_position = SimplePosition(self.calendar.start, 0)
        self.calendar.fresh_start()
        for date in tqdm(self.calendar):
            signal = self.signal_provider.get_signal_on_date(date)
            closing_signal = self.closing_signal_provider.get_signal_on_date(date)
            self.current_position.set_date(date)
            if signal == 1 and self.current_position.shares == 0:
                self.current_position.set_shares(1)
            elif signal == -1 and self.current_position.shares == 0:
                self.current_position.set_shares(-1)
            elif signal == 1 and self.current_position.shares == -1:
                self.current_position.set_shares(0)
            elif signal == -1 and self.current_position.shares == 1:
                self.current_position.set_shares(0)
            else:
                pass
            
            if closing_signal == 1:
                self.current_position.set_shares(0)
            else:
                pass
            self.arhive_position()


class BacktestingCalendar:
    def __init__(self, dates):
        # turn the dates into a pandas index for easier get_loc
        if isinstance(dates, pd.DatetimeIndex) and dates.dtype == '<M8[ns]':
            dates = dates.date
        self.dates = pd.Index(dates)
        self.current_date = None
        self.current_index = None
        self.length = len(dates)
        self.start = dates[0]
        self.end = dates[-1]
        
    def set_current_date(self, date):
        if date in self.dates:
            self.current_date = date
            self.current_index = self.dates.get_loc(date)
        else:
            raise ValueError('Date not in the calendar')
        
    def get_offset_date(self, offset):
        return self.dates[self.current_index + offset]
    
    def get_current_date(self):
        return self.current_date
        
    def fresh_start(self):
        self.current_date = self.start
        self.current_index = 0

    def __str__(self) -> str:
        return f'BacktestingCalendar({self.start}, {self.end}), current date: {self.current_date}'
    
    def __repr__(self) -> str:
        return str(self)

    def __iter__(self):
        if self.current_date is None or self.current_index is None:
            raise ValueError('Current date / index is not set')
        return self
    
    def __next__(self):
        if self.current_index < self.length:
            self.current_date = self.dates[self.current_index]
            self.current_index += 1
            return self.current_date
        else:
            self.current_date = None
            self.current_index = None
            raise StopIteration
