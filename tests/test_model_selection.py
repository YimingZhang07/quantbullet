import numpy as np
import pandas as pd
import unittest

try:
    from quantbullet.model_selection import TimeSeriesDailyRollingSplit, time_series_cv_predict
    from quantbullet.model import LastValueEstimator
except ImportError:
    TimeSeriesDailyRollingSplit = None
    time_series_cv_predict = None
    LastValueEstimator = None

@unittest.skipIf(TimeSeriesDailyRollingSplit is None, "TimeSeriesDailyRollingSplit not available")
@unittest.skipIf(time_series_cv_predict is None, "time_series_cv_predict not available")
@unittest.skipIf(LastValueEstimator is None, "LastValueEstimator not available")
class TestModelSelection( object ):
    def setup_class(self):
        self.df = pd.DataFrame({'date': ['2019-01-01', '2019-01-01', '2019-01-02', '2019-01-02', '2019-01-03', '2019-01-03'],
                                'value': [1, 2, 3, 4, 5, 6]})

    def test_time_series_daily_rolling_split(self):
        splitter = TimeSeriesDailyRollingSplit(min_train_size=1, max_train_size=None)
        splits = splitter.split(self.df)
        np.testing.assert_array_equal(next(splits)[0], np.array([0, 1]))
        np.testing.assert_array_equal(next(splits)[0], np.array([0, 1, 2, 3]))

    def test_time_series_cv_predict(self):
        model = LastValueEstimator(reference_column='date')
        splitter = TimeSeriesDailyRollingSplit(min_train_size=1, max_train_size=None)
        cv = list(splitter.split(self.df))
        y_pred = time_series_cv_predict(model, self.df, self.df['value'], cv)
        np.testing.assert_array_equal(y_pred, np.array([np.nan, np.nan, 1.5, 1.5, 3.5, 3.5]))