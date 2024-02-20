import numpy as np
import pandas as pd

def compute_ex_ante_volatility(daily_returns, com=60, annualization_factor=261):
    """
    Compute the ex ante annualized volatility for a series of daily returns.
    
    Parameters
    ----------
    daily_returns : pd.Series
        A series of daily returns.
    com : float, default=60
        The center of mass for the exponential moving average.
    annualization_factor : int, default=261
        
    Returns
    -------
    ex_ante_volatility : pd.Series
        The ex ante annualized volatility of the daily returns.
    """
    if isinstance(daily_returns, np.ndarray):
        daily_returns = pd.Series(daily_returns)
    if isinstance(daily_returns, list):
        daily_returns = pd.Series(daily_returns)
    
    # Calculate the exponentially weighted average return
    ewma_returns = daily_returns.ewm(com=com).mean()
    
    # Calculate the squared deviations from the exponentially weighted average return
    squared_deviations = (daily_returns - ewma_returns) ** 2
    
    # Calculate the exponentially weighted variance
    ewma_variance = squared_deviations.ewm(com=com).mean()
    
    # Annualize the variance
    annual_variance = ewma_variance * annualization_factor
    
    # Calculate the annualized volatility (standard deviation)
    ex_ante_volatility = np.sqrt(annual_variance)
    
    return ex_ante_volatility

    
def generate_ts_momentum_signal(returns, k = 5):
    """generate a time series momentum signal
    
    Look at the past k days of returns and generate a signal based on the sign of the sum of the returns
    
    Notes
    -----
    To avoid look-ahead bias, the look back period is from t-1 to t-k
    
    Parameters
    ----------
    returns : pd.Series
        A series of returns, assuming that the returns are already in log form
    k : int, default=5
        The number of days to look back.
        
    Returns
    -------
    signals : pd.Series
        A series of signals, where 1 indicates a long position, -1 indicates a short position, and 0 indicates no position
    """
    shifted_returns = returns.shift(1)
    rolling_sum = shifted_returns.rolling(window = k).sum()
    signals = np.sign(rolling_sum)
    return signals
    