"""
Module: quantbullet.backtest.backtest
"""


import unittest
import datetime
from collections import defaultdict

from quantbullet.backtest import Position, Account

class TestPosition(unittest.TestCase):
    def test_basic_calculation(self):
        position = Position(ticker='AAPL', shares=100, price=100, timestamp=datetime.date(2020, 1, 1))
        position.trade(shares=100, price=110, timestamp=datetime.date(2020, 1, 2))
        self.assertEqual(position.MarketValue, 22000)
        self.assertEqual(position.PnL, 1000)
        position.trade(shares=50, price=105, timestamp=datetime.date(2020, 1, 3))
        self.assertEqual(position.MarketValue, 250 * 105)
        # the first trade earns 500 and the second trade losses 500
        self.assertEqual(position.PnL, 0)
        # closing position
        position.trade(shares=-250, price=105, timestamp=datetime.date(2020, 1, 4))
        self.assertEqual(position.MarketValue, 0)
        self.assertEqual(position.PnL, 0)
        # open position again
        position.trade(shares=100, price = 110, timestamp=datetime.date(2020, 1, 5))
        self.assertEqual(position.MarketValue, 11000)
        self.assertEqual(position.PnL, 0)
        # update market price
        position.updateMarketPrice(price=115, timestamp=datetime.date(2020, 1, 6))
        self.assertEqual(position.MarketValue, 11500)
        self.assertEqual(position.PnL, 500)

    def test_history(self):
        position = Position(ticker='AAPL', shares=100, price=100, timestamp=datetime.date(2020, 1, 1))
        position.trade(shares=100, price=110, timestamp=datetime.date(2020, 1, 2))
        position.trade(shares=50, price=105, timestamp=datetime.date(2020, 1, 3))
        position.trade(shares=-250, price=105, timestamp=datetime.date(2020, 1, 4))
        position.trade(shares=100, price = 110, timestamp=datetime.date(2020, 1, 5))
        position.updateMarketPrice(price=115, timestamp=datetime.date(2020, 1, 6))
        self.assertEqual(len(position.getHistory()), 6)
        # add more history on same day and check if the length is correct
        position.trade(shares=100, price=120, timestamp=datetime.date(2020, 1, 6))
        self.assertEqual(len(position.getHistory()), 7)
        # check the type of the history
        self.assertEqual(type(position.getHistory()), list)
        self.assertIsInstance(position.getHistory()[0], tuple)
        # check the dataframae
        df = position.getHistoryDataFrame()
        self.assertEqual(df.shape, (7, 7))