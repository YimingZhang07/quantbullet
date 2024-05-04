import pandas as pd
import numpy as np
import math
import unittest

from quantbullet.research.momentum import (
    TimeSeriesMomentumSignal,
    ExAnteVolatilityEstimator,
)


class Test_Momentum(unittest.TestCase):
    def test_generate_ts_momentum_signal(self):
        date_range = pd.date_range(start="1/1/2020", periods=5, freq="D")
        returns = pd.Series([1, 0, 1, -1, -2], index=date_range)
        signal_object = TimeSeriesMomentumSignal.FromSeries(returns, k=3)
        signals = signal_object.to_list()
        np.testing.assert_array_equal(signals, [np.nan, np.nan, 1, 0, -1])

    def test_ExAnteVolatilityEstimator(self):
        date_range = pd.date_range(start="1/1/2020", periods=7, freq="D")
        returns = pd.Series([0, 1, 2, 1, 3, 2, 1], index=date_range)
        estimator = ExAnteVolatilityEstimator.FromSeries(
            returns, com=3, annualization_factor=1
        )

        # underlying approach
        ewma_returns = returns.ewm(com=3).mean()
        squared_deviations = (returns - ewma_returns) ** 2
        ewma_vol = np.sqrt(squared_deviations.ewm(com=3).mean()).tolist()
        np.testing.assert_almost_equal(estimator.values, ewma_vol)
