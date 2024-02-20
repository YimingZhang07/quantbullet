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