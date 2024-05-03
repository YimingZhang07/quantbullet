"""
Module:     backtest.backtest
"""


import unittest
import datetime

from quantbullet.backtest import Position, Account

class TestPosition(unittest.TestCase):
    def test_basic_calculation(self):
        position = Position(ticker='AAPL', shares=100, market_price=100, timestamp=datetime.date(2020, 1, 1))
        position.trade(shares=100, price=110, timestamp=datetime.date(2020, 1, 2))
        self.assertEqual(position.MarketValue, 22000)
        self.assertEqual(position.PnL, 1000)
        position.trade(shares=50, price=105, timestamp=datetime.date(2020, 1, 3))
        self.assertEqual(position.MarketValue, 5250)
        self.assertEqual(position.PnL, 250)
