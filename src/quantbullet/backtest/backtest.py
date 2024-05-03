import pandas as pd
import numpy as np
from collections import defaultdict
from operator import itemgetter

def take_position_snapshot(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        timestamp = kwargs.get('timestamp', None)
        if timestamp is not None:
            comment = f"{func.__name__}"
            self.trackPosition(timestamp, comment=comment)
        else:
            raise ValueError("No timestamp is given to track the positions.")
        return result
    return wrapper

class Position:
    """Position class to store position information"""

    def __init__(self, ticker='MISSING', shares=None, avg_cost=None, market_price=None, timestamp=None):
        self.ticker = ticker
        self.shares = shares
        self.market_price = market_price
        if avg_cost is None:
            self.avg_cost = market_price
        self._position_history = defaultdict(set)
        self._latest_snapshot = None

    def __str__(self):
        return (f"{self.ticker}: {self.shares} shares, {self.avg_cost} avg cost, "
                f"{self.market_price} market price, {self.market_value} market value, "
                f"{self.PnL} PnL")

    def __repr__(self):
        return f"<Position {self.ticker}>"

    def as_dict(self):
        return {
            'ticker': self.ticker,
            'shares': self.shares,
            'avg_cost': self.avg_cost,
            'market_price': self.market_price,
            'market_value': self.market_value,
            'PnL': self.PnL
        }

    @property
    def market_value(self):
        return self.shares * self.market_price

    @property
    def MarketValue(self):
        return self.market_value
    
    @property
    def AvgCost(self):
        return self.avg_cost

    @property
    def PnL(self):
        return self.market_value - self.avg_cost * self.shares

    @take_position_snapshot
    def trade(self, shares, price, timestamp):
        """Trade shares of a position"""
        if self.shares + shares != 0:
            self.avg_cost = (self.shares * self.avg_cost + shares * price) / (self.shares + shares)
        else:
            self.avg_cost = 0.
            self.shares = 0
        self.market_price = price

    @take_position_snapshot
    def updateMarketPrice(self, price, timestamp):
        """Update market price per share"""
        self.market_price = price

    @take_position_snapshot
    def updateShares(self, shares, timestamp):
        """Update shares ONLY. This is used for cash position.
        Note: This will replace the current shares with the new shares instead of adding."""
        self.shares = shares

    def getHistory(self):
        return self._position_history
    
    def getHistoryDataframe(self):
        records = [
            snapshot for timestamp, snapshots in self.getHistory().items() for snapshot in snapshots
        ]
        return pd.DataFrame.from_records(records, columns=['shares', 'avg_cost', 'market_price', 'market_value', 'PnL', 'comment'])

    def trackPosition(self, timestamp, comment=""):
        self._latest_snapshot = (self.shares, self.avg_cost, self.market_price, self.market_value, self.PnL, comment)
        self._position_history[timestamp].add(self._latest_snapshot)

class Account:
    """Account class to store a portfolio of positions with cash.
    
    Notes
    -----
    Within each account, it is designed that each ticker can have only one position
    
    """
    
    def __init__(self, name='default', init_cash=0, timestamp='2000-01-01'):
        self._name = name
        self._init_cash = init_cash
        self._cur_position = defaultdict(Position)
        self._cur_position['CASH'] = Position(
            ticker="CASH", shares=init_cash, price=1, timestamp=timestamp)
    
    @property
    def Name(self):
        return self._name
    
    @property
    def CurrentPosition(self):
        return self._cur_position
    
    @property
    def PositionTickers(self):
        return list(self.CurrentPosition.keys())
    
    def getPositionForTicker(self, ticker):
        return self.CurrentPosition[ticker]
    
    @property
    def CashAmount(self):
        return self.CurrentPosition['CASH'].market_value
    
    @property
    def MarketValue(self):
        return sum([position.market_value for position in self.CurrentPosition.values()])
    
    def payCash(self, amount, timestamp):
        """Update cash position
        Args:
            amount (float): amount to update
        """
        after_cash = self._cur_position['CASH'].shares + amount
        self._cur_position['CASH'].updateShares(after_cash, timestamp=timestamp)
    
    def trade(self, ticker, shares, price, timestamp):
        if ticker not in self._cur_position:
            self._cur_position[ticker] = Position(
                ticker=ticker, shares=shares, price=price, timestamp=timestamp)
        else:
            self._cur_position[ticker].trade(shares, price, timestamp=timestamp)
            self.payCash(-shares * price, timestamp)
    
    def getPositionSnapshot(self, need_sort=False):
        """Get explicit list of positions and values to create snapshot
        Note: Implemented using map and itemgetter which is much faster than creating dataframe.
        
        Returns:
        
        """
        snapshot = list(map(itemgetter('ticker', 'shares', 'avg_cost', 'market_price', 'market_value', 'PnL'), 
                            [pos.as_dict() for pos in self.CurrentPosition.values()]))
        
        if need_sort:
            snapshot.sort(key=itemgetter(4), reverse=True)
        
        # map the function returns a map object, which is an iterator. use it if you don't need to store the result in memory
        total_PnL = sum(map(itemgetter(5), snapshot))
        total_market_value = sum(map(itemgetter(4), snapshot))
        
        return snapshot, total_market_value, total_PnL
