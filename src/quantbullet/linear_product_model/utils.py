import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression

def init_betas_by_response_mean(X, target_mean):
    """
    Initialize regression coefficients so that the mean prediction matches the target mean.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix including intercept column if applicable.
    target_mean : float
        Desired mean of the predictions.
        
    Returns
    -------
    np.ndarray
        Initialized regression coefficients.
    """
    c = X.mean(axis=0)
    denom = np.dot(c, c)
    if denom == 0:
        raise ValueError("Column means are all zero")
    return (target_mean / denom) * c

def fit_logistic_no_intercept(X, y):
    r = LogisticRegression( fit_intercept=False, solver='lbfgs' ).fit(X, y)
    return r.coef_.ravel()

def log_loss(y_hat, y):
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)

