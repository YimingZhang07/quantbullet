import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .validation import are_only_values_in_series

def plot_shared_x(
    x,
    y1,
    y2,
    color1="tab:red",
    color2="tab:blue",
    xlabel="x",
    ylabel1="y1",
    ylabel2="y2",
):
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, color=color1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color1)
    ax1.tick_params("y", colors=color1)

    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=color2)
    ax2.set_ylabel(ylabel2, color=color2)
    ax2.tick_params("y", colors=color2)

    fig.tight_layout()
    return fig, ax1, ax2


def plot_price_logret_volatility(
    price_series,
    log_returns,
    volatility,
    label_price="Price",
    label_volatility="Volatility",
    label_log_returns="Log Returns",
    figsize=(10, 8),
):
    fig, axs = plt.subplots(
        3, 1, figsize=figsize, sharex=True
    )  # Share the x-axis (time)

    axs[0].plot(price_series, color="green", label=label_price)
    axs[0].set_title(label_price)
    axs[0].legend()

    axs[1].plot(volatility, label=label_volatility)
    axs[1].set_title(label_volatility)
    axs[1].legend()

    axs[2].plot(log_returns, color="orange", label=label_log_returns)
    axs[2].set_title(label_log_returns)
    axs[2].legend()

    plt.xlabel("Time")  # Common X-axis label
    plt.tight_layout()  # Adjust layout to not overlap
    plt.show()
    return fig, axs

def plot_price_with_signal(prices, signals, figsize=(14, 7)):
    """plot the price chart with long/short signals
    
    Parameters
    ----------
    prices : pd.Series
        A series of prices
    signals : pd.Series
        A series of signals, where 1 indicates a long position, -1 indicates a short position, and 0 indicates no position
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes._subplots.AxesSubplot
    """
    
    if not isinstance(prices, pd.Series):
        raise ValueError("prices must be a pandas Series with a datetime index")
    
    if not isinstance(signals, pd.Series):
        raise ValueError("prices must be a pandas Series with a datetime index")

    if not are_only_values_in_series(signals, [-1, 0, 1, np.nan]):
        raise ValueError("signals must contain only -1, 0, or 1")
    
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)

    # Price data
    ax.plot(prices.index, prices, label='Price', alpha=0.7)

    # Buy signals
    buy_signals = signals[signals == 1]
    ax.scatter(buy_signals.index, prices.loc[buy_signals.index], label='Buy', color='green', marker='^', alpha=1)

    # Sell signals
    sell_signals = signals[signals == -1]
    ax.scatter(sell_signals.index, prices.loc[sell_signals.index], label='Sell', color='red', marker='v', alpha=1)

    # Customization and labels
    ax.set_title('Price Chart with Long/Short Signals')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

    # plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig, ax
