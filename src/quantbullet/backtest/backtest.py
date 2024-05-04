"""
Tests: tests.backtest.test_backtest
"""


import pandas as pd
import numpy as np
import datetime
from collections import defaultdict
from operator import itemgetter


def take_position_snapshot(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        timestamp = kwargs.get( 'timestamp', None )
        if timestamp is not None:
            comment = f"{func.__name__}"
            self.trackPosition(timestamp=timestamp, comment=comment)
        else:
            raise ValueError("No timestamp is given to track the positions.")
        return result
    return wrapper

class Position:
    """Position class to store position information
    """
    # using slots to save memory and speed up the attribute access
    # __slots__ = ['ticker', 'shares', 'avg_cost', 'market_price', 'market_value', 'PnL']

    @take_position_snapshot
    def __init__(self, ticker='MISSING', shares=None, price=None, timestamp=None):
        self.ticker = ticker
        self.shares = shares
        self.avg_cost = price
        self.market_price = price
        # to track the position history
        self._position_history = list()
        self._lastest_snapshot = None

    def __str__(self) -> str:
        return f'{self.ticker}: {self.shares} shares, {self.avg_cost} avg cost, {self.market_price} market price, {self.market_value} market value, {self.PnL} PnL'

    def __repr__(self) -> str:
        return f'{self.ticker}: {self.shares} shares, {self.avg_cost} avg cost, {self.market_price} market price, {self.market_value} market value, {self.PnL} PnL'
    
    def as_dict(self) -> dict:
        return{
            'ticker'        : self.ticker,
            "shares"        : self.shares,
            "avg_cost"      :self.avg_cost,
            "market_price"  : self.market_price,
            "market_value"  : self.market_value,
            "PnL"           : self.PnL
        }
    
    @property
    def market_value(self):
        return self.shares * self.market_price
    
    @property
    def MarketValue(self):
        return self.market_value
    
    @property
    def PnL(self):
        return self.market_value - self.avg_cost * self.shares

    @take_position_snapshot
    def trade(self, shares, price, timestamp):
        """Trade shares of a position

        Args:
            shares (float): number of shares
            price (float): price per share
        """
        if self.shares + shares != 0:
            self.avg_cost = (self.avg_cost * self.shares +
                            shares * price) / (self.shares + shares)
            self.shares += shares
        else:
            self.avg_cost = 0.
            self.shares = 0
        self.market_price = price

    @take_position_snapshot
    def updateMarketPrice(self, price, timestamp):
        """Update market price

        Args:
            price (float): price per share
        """
        self.market_price = price

    @take_position_snapshot
    def updateShares(self, shares, timestamp):
        """Update shares ONLY. This is used for cash position.

        Note: This will replace the current shares with the new shares instead of adding.

        Args:
            shares (float): number of shares
        """
        self.shares = shares

    def getHistory(self):
        return self._position_history
    
    # def getHistoryDataFrame(self):
    #     records = [
    #         snapshot for timestamp, snapshots in self.getHistory().items() for snapshot in snapshots
    #     ]
    #     return pd.DataFrame.from_records(records, columns=['shares', 'avg_cost', 'market_price', 'market_value', 'pnl', 'comment'])
    
    def getHistoryDataFrame(self):
        return pd.DataFrame(self._position_history, columns=['timestamp', 'shares', 'avg_cost', 'market_price', 'market_value', 'pnl', 'comment'])

    def trackPosition(self, timestamp, comment=""):
        self._lastest_snapshot = (timestamp, self.shares, self.avg_cost, self.market_price, self.market_value, self.PnL, comment)
        # self._position_history[timestamp].add(self._lastest_snapshot)
        self._position_history.append(self._lastest_snapshot)

class Account:
    """Account class to store a portolio of positions with cash.

    Notes
    -----
    Within each account, it is designed that each ticker can have only one position

    """

    def __init__(self, name='default', init_cash=0, timestamp='2000-01-01'):
        self._name = name
        self._initial_cash = init_cash
        self._cur_position = defaultdict(Position)
        self._cur_position['CASH'] = Position(
            ticker='CASH', shares=init_cash, price=1, timestamp=timestamp)
    
    @property
    def Name(self):
        return self._name

    @property
    def CurrentPosition(self):
        return self._cur_position
    
    @property
    def PositionTickers(self):
        return list(self.CurrentPosition.keys()).remove('CASH')

    def getPositionForTicker(self, ticker):
        return self.CurrentPosition[ticker]

    @property
    def CashAmount(self):
        return self.CurrentPosition['CASH'].market_value
    
    @property
    def PnL(self):
        return sum([position.PnL for position in self.CurrentPosition.values()])
    
    @property
    def MarketValue(self):
        return sum([position.market_value for position in self.CurrentPosition.values()])

    def _payCash(self, amount, timestamp):
        """Update cash position"""
        after_cash = self._cur_position['CASH'].shares + amount
        self._cur_position['CASH'].updateShares(after_cash, timestamp=timestamp)

    def trade(self, ticker, shares, price, timestamp):
        if ticker not in self._cur_position:
            self._cur_position[ticker] = Position(
                ticker=ticker, shares=shares, price=price, timestamp=timestamp)
            self._payCash(-shares * price, timestamp)
        else:
            self._cur_position[ticker].trade(shares, price, timestamp=timestamp)
            self._payCash(-shares * price, timestamp)
            
    def updateMarketPrice(self, ticker, price, timestamp):
        if ticker in self._cur_position:
            self._cur_position[ticker].updateMarketPrice(price, timestamp=timestamp)
        else:
            raise ValueError(f"Ticker {ticker} not found in the account")

    def getPositionSnapshot(self, need_sort=False):
        """
        Get the snapshot of the current positions
        
        Parameters
        ----------
        need_sort : bool, optional
            Sort the snapshot by market value in descending order, by default False
            
        Returns
        -------
        tuple
            snapshot, total_market_value, total_PnL
        """
        snapshot = list(map(itemgetter("ticker", "shares", "avg_cost", "market_price", "market_value", "PnL"),
                            [pos.as_dict() for pos in self.CurrentPosition.values()]))

        if need_sort:
            snapshot.sort(key=itemgetter(4), reverse=True)
        # the map function returns a map object, which is an iterator. use it if you don't need to store the result in memory
        # tickers = list(map(itemgetter(0), res))
        total_market_value = sum(map(itemgetter(4), snapshot))
        total_PnL = sum(map(itemgetter(5), snapshot))
        return snapshot, total_market_value, total_PnL