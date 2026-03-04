import numpy as np
import numexpr as ne
from typing import Dict, Optional, Union, List, Sequence

def ols_normal_equation(X, y, ridge=1e-8, weights=None):
    """Solve the least squares problem using the normal equations with optional ridge regularization.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    ridge : float, optional
        Ridge regularization parameter. Default is 1e-8.

    Returns
    -------
    np.ndarray
        Estimated coefficients.
    """

    if weights is not None:
        sqrt_w = np.sqrt(weights)
        X = X * sqrt_w[:, None]
        y = y * sqrt_w

    XtX = X.T @ X
    if ridge > 0:
        XtX = XtX + ridge * np.eye(XtX.shape[0], dtype=XtX.dtype)
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    return beta

def ols_normal_equation_scaled(X, y, scale, weights=None, ridge=1e-8):
    """Solve scaled WLS in one pass: min ||sqrt(W) (diag(scale) X beta - y)||^2.

    Equivalent to ``ols_normal_equation(X * scale[:, None], y, weights=weights)``
    but builds only ONE n×p intermediate instead of two when weights are present.
    """
    d = scale ** 2
    if weights is not None:
        d = d * weights

    Xw = X * np.sqrt(d)[:, None]
    XtX = Xw.T @ Xw
    if ridge > 0:
        XtX[np.diag_indices_from(XtX)] += ridge

    rhs = scale * y
    if weights is not None:
        rhs = rhs * weights
    Xty = X.T @ rhs

    return np.linalg.solve(XtX, Xty)


def vector_product_numexpr_list(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """
    Compute element-wise product of a sequence of vectors using numexpr.
    
    Parameters
    ----------
    arrays : sequence of np.ndarray
        List/tuple of 1D numpy arrays (same length).
    
    Returns
    -------
    np.ndarray
        Element-wise product of all arrays.
    """
    if len(arrays) == 0:
        raise ValueError("No arrays provided!")

    expr = "*".join([f"x{i}" for i in range(len(arrays))])
    local_dict = {f"x{i}": arr for i, arr in enumerate(arrays)}

    return ne.evaluate(expr, local_dict=local_dict)

def vector_product_numexpr_dict_values(
    data: Dict[str, np.ndarray], 
    exclude: Optional[Union[str, List[str]]] = None
) -> np.ndarray:
    """
    Compute element-wise product of vectors in a dictionary, 
    with optional exclusion of some keys. Uses numexpr internally.
    """
    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    arrays = [v for k, v in data.items() if k not in exclude]
    if not arrays:
        ref = next(iter(data.values()))
        return np.ones(len(ref), dtype=ref.dtype)

    return vector_product_numexpr_list(arrays)