import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

def plot_diagnostics(series, title="Time Series Diagnostics"):
    """Plot diagnostics for a time series
    
    Parameters
    ----------
    series : array-like
        The time series data

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    # ACF plot
    plot_acf(series, ax=axes[0, 0])
    axes[0, 0].set_title(f"{title} - Autocorrelation Function (ACF)")
    # axes[0, 0].axhline(y=0, linestyle='--', color='gray')
    # axes[0, 0].axhline(y=1.96/np.sqrt(len(series)), linestyle='--', color='gray')
    # axes[0, 0].axhline(y=-1.96/np.sqrt(len(series)), linestyle='--', color='gray')
    axes[0, 0].set_ylim(-0.2, 1.2)
    
    # Q-Q plot
    sm.qqplot(series, line='s', ax=axes[0, 1])
    axes[0, 1].set_title(f"{title} - Q-Q Plot vs. Normal")
    
    # Time series plot
    fig.add_subplot(2, 1, 2)
    plt.plot(series)
    plt.title(f"{title} - Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xlim(0, len(series))
    plt.ylim(min(series), max(series))
    
    # Remove the empty axes
    fig.delaxes(axes[1, 0])
    fig.delaxes(axes[1, 1])
    
    plt.tight_layout()
    plt.show()


def stationary_adf_test(series):
    """Augmented Dickey-Fuller test for stationarity
    
    Parameters
    ----------
    series : array-like
        The time series data

    Returns
    -------
    p_value : float
        The p-value of the test
    """
    result = adfuller(series)
    adf_statistic = result[0]
    p_value = result[1]
    used_lag = result[2]
    nobs = result[3]
    critical_values = result[4]
    
    print(f"ADF Statistic: {adf_statistic}")
    print(f"p-value: {p_value}")
    print(f"Used lag: {used_lag}")
    print(f"Number of observations: {nobs}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"   {key}: {value}")

    return p_value

def autocorr_ljungbox_test(series, lags=10):
    """Ljung-Box test for autocorrelation
    
    Parameters
    ----------
    series : array-like
        The time series data
    lags : int
        The number of lags to consider

    Returns
    -------
    lb_test : DataFrame
        A DataFrame containing the test statistics and p-values
    """
    print("Null hypothesis: No autocorrelation up to lag k")
    lb_test = acorr_ljungbox(series, lags=lags, return_df=True)
    return lb_test