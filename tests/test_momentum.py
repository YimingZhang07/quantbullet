import pandas as pd
import unittest

from quantbullet.research.momentum import (
    TimeSeriesMomentumSignal
)


class Test_Momentum(unittest.TestCase):
    def test_generate_ts_momentum_signal(self):
        date_range = pd.date_range(start="1/1/2020", periods=5, freq="D")
        returns = pd.Series([1, 0, 1, -1, -2], index=date_range)
        signal_object = TimeSeriesMomentumSignal.FromSeries(returns, k=3)
        singals = signal_object.to_list()
        self.assertListEqual(singals, [float('nan'), float('nan'), 1, 0, -1])