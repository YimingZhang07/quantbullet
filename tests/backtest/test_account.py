"""
Module: quantbullet.backtest.backtest
"""

import unittest
import datetime

from quantbullet.backtest import Account

class TestAccount(unittest.TestCase):
    def setUp(self):
        self.account = Account(name='test', init_cash=10000, timestamp=datetime.date(2020, 1, 1))
        self.account.trade(ticker='AAPL', shares=100, price=100, timestamp=datetime.date(2020, 1, 1))
        # add more trades
        self.account.trade(ticker='AAPL', shares=100, price=110, timestamp=datetime.date(2020, 1, 2))
        # add more tickers
        self.account.trade(ticker='MSFT', shares=50, price=200, timestamp=datetime.date(2020, 1, 3))
    
    def test_trades(self):
        self.assertEqual(self.account.CashAmount, -11000 - 10000)
        self.assertEqual(self.account.MarketValue, 11000)
        self.assertEqual(self.account.PnL, 1000)
        
    def test_getPositionSnapshot(self):
        snapshot, total_market_value, total_PnL = self.account.getPositionSnapshot()
        self.assertEqual(len(snapshot), 3)
        self.assertEqual(total_market_value, 11000)
        self.assertEqual(total_PnL, 1000)
        
    def test_updateMarketPrice(self):
        self.account.updateMarketPrice(ticker='AAPL', price=120, timestamp=datetime.date(2020, 1, 4))
        self.assertEqual(self.account.MarketValue, 13000)
        self.assertEqual(self.account.PnL, 3000)