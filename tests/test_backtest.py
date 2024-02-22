import unittest
import datetime
from quantbullet.utils.backtest import *

class TestBacktestUtils(unittest.TestCase):
    def test_SimpleDataProvider(self):
        pass

    def test_SimpleSignalProvider(self):
        signals = pd.Series([1, 0, -1, 0, 1, -1], index=pd.date_range('2021-01-01', periods=6))
        signal_provider = SimpleSignalProvider(signals)
        self.assertEqual(signal_provider.get_signal_on_date(datetime.date(2021, 1, 1)), 1)