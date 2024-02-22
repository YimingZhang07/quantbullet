import pandas as pd
import numpy as np

def compute_log_returns(prices):
    """compute the log returns of a series of prices
    
    Parameters
    ----------
    prices : pd.Series
        A series of prices
    
    Returns
    -------
    log_returns : np.ndarray
        The log returns of the prices
    """
    # return different types given types of input
    # e.g. if prices is a Series with a datetime index, the result is a Series with a datetime index
    if isinstance(prices, pd.Series):
        return np.log(prices / prices.shift(1))
    elif isinstance(prices, np.ndarray) or isinstance(prices, list):
        prices = pd.Series(prices)
        return np.log(prices / prices.shift(1)).values
    else:
        raise ValueError("Invalid input type")
    
def compute_sharpe_ratio(returns, annualization_factor=252):
    return np.sqrt(annualization_factor) * returns.mean() / returns.std()

def compute_max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    return (cum_returns / cum_returns.cummax() - 1).min()

def print_metrics(returns, annualization_factor=252):
    print(f'Annualized Return: {returns.mean() * annualization_factor:.2%}')
    print(f'Annualized Volatility: {returns.std() * np.sqrt(annualization_factor):.2%}')
    print(f'Sharp Ratio: {compute_sharpe_ratio(returns, annualization_factor):.2f}')
    print(f'Max Drawdown: {compute_max_drawdown(returns):.2%}')
    print(f'Number of Trades: {returns[returns != 0].count()}')