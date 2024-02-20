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