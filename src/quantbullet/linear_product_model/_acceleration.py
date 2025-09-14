import numpy as np
import numexpr as ne
from typing import Dict, Optional, Union, List, Sequence

def ols_normal_equation(X, y, ridge=1e-8):
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
    XtX = X.T @ X
    if ridge > 0:
        XtX = XtX + ridge * np.eye(XtX.shape[0], dtype=XtX.dtype)
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    return beta

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
        raise ValueError("No arrays left after exclusion!")

    return vector_product_numexpr_list(arrays)