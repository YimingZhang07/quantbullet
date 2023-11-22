import pytest
import numpy as np
import pandas as pd
from quantbullet.model_selection import TimeSeriesDailyRollingSplit

class TestModelSelection( object ):
    def setup_class(self):
        pass

    def test_time_series_daily_rolling_split(self):
        df = pd.DataFrame({'date': ['2019-01-01', '2019-01-01', '2019-01-02', '2019-01-02', '2019-01-03', '2019-01-03'],
                           'value': [1, 2, 3, 4, 5, 6]})
        splitter = TimeSeriesDailyRollingSplit(min_train_size=1, max_train_size=None)
        splits = splitter.split(df)
        np.testing.assert_array_equal(next(splits)[0], np.array([0, 1]))
        np.testing.assert_array_equal(next(splits)[0], np.array([0, 1, 2, 3]))
