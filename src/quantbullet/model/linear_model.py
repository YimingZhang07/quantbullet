import pandas as pd
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
#     """
#     Fit an AR(1) model to a single time series.

#     Parameters:
#     -----------
#         series (numpy.ndarray or pandas.Series): The time series data.

#     Returns:
#     --------
#         statsmodels.tsa.arima.model.ARIMAResultsWrapper:
#             The fitted AR(1) model.
#     """
#     # Fit AR(1) model
#     ar1_model = sm.tsa.ARIMA(series, order=order).fit()

#     return ar1_model
    raise NotImplementedError("This function is not implemented yet.")

def ar_ols(series, lag_order=1):
    """
    Fit an autoregressive (AR) model to a single time series using OLS regression.

    Parameters:
    -----------
        series (numpy.ndarray or pandas.Series): The time series data.
        lag_order (int): The order of the autoregressive model (number of lags).

    Returns:
    --------
        numpy.ndarray: An array containing the autoregressive coefficients.
    """
    # Create lagged versions of the series
    lagged_data = pd.DataFrame({f"Lag_{i}": series.shift(i) for i in range(1, lag_order + 1)})
    
    # Concatenate lagged data with the original series
    lagged_data['Original'] = series
    
    # Drop missing values introduced by the lagging process
    lagged_data = lagged_data.dropna()
    
    # Extract independent variables (lagged series)
    X = lagged_data.iloc[:, :-1]
    
    # Add constant term
    X = sm.add_constant(X)
    
    # Dependent variable (original series)
    y = lagged_data['Original']
    
    # Fit OLS regression model
    model = sm.OLS(y, X).fit()
    return model