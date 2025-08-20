import numpy as np
from scipy.optimize import approx_fprime
from scipy.optimize import minimize
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

def _clipped_ce_and_grad(beta, X, y, eps=1e-6, l2=0.0):
    """
    Cross-entropy on p = clip(X @ beta, eps, 1-eps).
    Returns (loss, grad) with grad matching the piecewise derivative described.
    """
    z = X @ beta                          # (n,)
    p = np.clip(z, eps, 1.0 - eps)        # (n,)

    # ----- Loss -----
    # CE(p,y) = -[ y log p + (1-y) log(1-p) ]
    ce = - (y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    if l2:
        ce += 0.5 * l2 * np.dot(beta, beta)

    # ----- Gradient wrt beta -----
    # dCE/dp = (p - y) / (p(1-p))
    dCE_dp = (p - y) / (p * (1.0 - p))

    # dp/dz = 1 on the interior, 0 when clipped
    interior = (z > eps) & (z < 1.0 - eps)
    dL_dz = np.zeros_like(z)
    dL_dz[interior] = dCE_dp[interior]  # chain rule: dL/dz = dCE/dp * dp/dz

    grad = X.T @ dL_dz
    if l2:
        grad += l2 * beta

    return ce, grad

def minimize_clipped_cross_entropy_loss(X, y, beta0=None, eps=1e-6, l2=0.0, tol=1e-8, maxiter=10_000):
    """
    Fit a linear model by minimizing clipped cross-entropy loss. y should be in {0,1}.
    
    The model form is p = clip(X @ beta, eps, 1-eps). and the loss is
    the cross-entropy between p and y, plus optional L2 regularization.
    """
    n_features = X.shape[1]
    if beta0 is None:
        beta0 = init_betas_by_response_mean(X, np.mean(y))
    elif np.isscalar(beta0):
        beta0 = np.full(n_features, beta0)
    else:
        beta0 = np.asarray(beta0, dtype=float)
        if beta0.shape != (n_features,):
            raise ValueError(f"beta0 must be a scalar or a 1D array of shape ({n_features},)")
    obj = lambda b: _clipped_ce_and_grad(b, X, y, eps=eps, l2=l2)
    res = minimize(obj, beta0, method="L-BFGS-B", jac=True, tol=tol,
                   options={"maxiter": maxiter})
    return res.x, res

def estimate_ols_beta_se(X, y, beta):
    """Estimate the standard error of OLS coefficients."""
    n = X.shape[0]
    residuals = y - X @ beta
    sigma_squared = np.sum(residuals**2) / (n - X.shape[1])
    var_beta = sigma_squared * np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(var_beta))

def estimate_ols_beta_se_with_scalar_vector(X, y, beta, scalar_vector):
    """Assumes a linear term y = X @ beta and MSE loss function"""
    if not scalar_vector.ndim == 1:
        raise ValueError("scalar_vector must be 1-dimensional")
    if scalar_vector.shape[0] != X.shape[0]:
        raise ValueError("scalar_vector length must match number of columns in X")
    cX = X * scalar_vector[:, None]  # Scale X by scalar_vector
    return estimate_ols_beta_se(cX, y, beta)
    
def estimate_logistic_beta_se( X, beta ):
    """Assumes a logistic term y = sigmoid(X @ beta) and binary cross-entropy loss function.
    beta is supposed to be fitted.
    """
    p = 1 / (1 + np.exp(-X @ beta))
    w = p * (1 - p)
    var_beta = np.linalg.inv(X.T @ (w[:, None] * X))
    return np.sqrt(np.diag(var_beta))

####################################################################################################
# Start
# Below is for the model that has the form of y = X @ beta; and the loss function is cross-entropy.
####################################################################################################

def clipped_sum_cross_entropy_loss(y_hat, y, eps=1e-6):
    """Cross entropy loss for clipped( y_hat ) and y"""
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def linear_clipped_sum_cross_entropy_loss( X, y, beta, eps=1e-6 ):
    """Cross entropy loss for clipped( X @ beta ) and y"""
    y_hat = X @ beta
    return clipped_sum_cross_entropy_loss(y_hat, y, eps=eps)

def estimate_linear_cross_entropy_beta_hessian( X, y, beta, eps=1e-6 ):
    """Assumes a linear term y = X @ beta and cross-entropy loss function.

    There is no sigmoid function applied to the output, and we just clip the output, and feed to the cross-entropy loss function directly.
    """
    y_hat = X @ beta
    y_hat = np.clip(y_hat, eps, 1 - eps)
    ##### original formula #####
    # w_num = y_hat * (1 - y_hat) - (y_hat - y) * (1 - 2 * y_hat)
    # w_denom = ( y_hat * (1 - y_hat) ) ** 2
    # w = w_num / w_denom
    ##### simplified formula #####
    w = 1 / ( 1 - y - y_hat ) ** 2

    ##### formula if we treat diag(w) as a scaler of identity matrix using expected values of w #####
    # w = 1 / np.mean( ( 1 - y - y_hat ) ** 2 ) * np.ones_like(w)
    H = X.T @ (w[:, None] * X)
    return H

def estimate_linear_cross_entropy_beta_se( X, y, beta, eps=1e-6 ):
    H = estimate_linear_cross_entropy_beta_hessian( X, y, beta, eps=eps )
    return np.sqrt(np.diag(np.linalg.inv(H)))

####################################################################################################
# End
####################################################################################################

def numerical_hessian(fun, beta, eps=1e-5, *args):
    n = len(beta)
    H = np.zeros((n, n))
    for i in range(n):
        beta_up = beta.copy(); beta_up[i] += eps
        beta_dn = beta.copy(); beta_dn[i] -= eps
        grad_up = approx_fprime(beta_up, fun, eps, *args)
        grad_dn = approx_fprime(beta_dn, fun, eps, *args)
        H[:, i] = (grad_up - grad_dn) / (2 * eps)
    return H