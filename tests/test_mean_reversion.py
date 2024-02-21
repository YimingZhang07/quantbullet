import unittest
import numpy as np
import pandas as pd
from quantbullet.research.mean_reversion import generate_band_based_signal

class TestMeanReversion(unittest.TestCase):
    def test_generate_signal_from_bands(self):
        date_range = pd.date_range(start='1/1/2020', periods=7, freq='D')
        prices = pd.Series([1, 2, 3, 4, 5, 6, 2], index=date_range)
        upper_band = pd.Series([np.nan, 2, 3, 4, 5, 5, 5], index=date_range)
        lower_band = pd.Series([0, 1, 2, 3, 4, 4, 4], index=date_range)
        signal = generate_band_based_signal(prices, upper_band, lower_band)
        expected_signal = pd.Series([np.nan, -1, -1, -1, -1, -1, 1], index=date_range)
        pd.testing.assert_series_equal(signal, expected_signal)