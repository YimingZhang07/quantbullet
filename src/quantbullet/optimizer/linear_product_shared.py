import numpy as np

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