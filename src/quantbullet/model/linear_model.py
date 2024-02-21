import statsmodels.api as sm

def ols_regression(X, y):
    """
    Perform Ordinary Least Squares (OLS) regression.

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Independent variables.
    y : pandas.Series or numpy.ndarray
        Dependent variable.

    Returns:
    --------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model.
    """
    X_with_constant = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X_with_constant).fit()
    return model

def arima(series, order=(1, 0, 0)):
    """
    Fit an AR(1) model to a single time series.

    Parameters:
    -----------
        series (numpy.ndarray or pandas.Series): The time series data.

    Returns:
    --------
        statsmodels.tsa.arima.model.ARIMAResultsWrapper:
            The fitted AR(1) model.
    """
    # Fit AR(1) model
    ar1_model = sm.tsa.ARIMA(series, order=order).fit()

    return ar1_model