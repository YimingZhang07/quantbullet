import numpy as np
import cvxpy as cp

def smooth_monotone_fit(xs, ys, weights=None, n_grid=200, alpha=10.0, increasing=True):
    """
    Fit a smooth, monotonic curve to (xs, ys) using convex optimization with a
    smoothness penalty on the second difference.

    Parameters
    ----------
    xs : array-like
        1D array of x-values (must be numeric and sorted or will be sorted internally).
    ys : array-like
        1D array of y-values, same length as xs.
    weights : array-like or None
        Optional non-negative weights; default = equal weights.
    n_grid : int
        Number of grid points for the fitted curve.
    alpha : float
        Smoothness strength. Larger → smoother curve; smaller → closer to data.
    increasing : bool
        True → enforce non-decreasing fit; False → non-increasing.

    Returns
    -------
    grid_x : np.ndarray
        Grid of x values where curve is evaluated.
    fitted_y : np.ndarray
        Fitted y values (monotone and smooth).
    """

    # Convert and sort data
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    n = len(xs)

    if weights is None:
        weights = np.ones_like(xs)
    weights = np.asarray(weights, dtype=float)
    weights /= np.sum(weights)

    # Create grid
    x_min, x_max = xs.min(), xs.max()
    grid_x = np.linspace(x_min, x_max, n_grid)

    # Build interpolation (A) matrix mapping grid values to data points
    idx = np.searchsorted(grid_x, xs, side='left').clip(1, len(grid_x) - 1)
    t = (xs - grid_x[idx - 1]) / (grid_x[idx] - grid_x[idx - 1])
    A = np.zeros((n, n_grid))
    for i in range(n):
        A[i, idx[i] - 1] = 1 - t[i]
        A[i, idx[i]] = t[i]

    # Smoothness (second difference) penalty matrix
    D2 = np.zeros((n_grid - 2, n_grid))
    for k in range(n_grid - 2):
        D2[k, k] = 1
        D2[k, k + 1] = -2
        D2[k, k + 2] = 1

    # Monotonicity (first difference) constraint
    D1 = np.zeros((n_grid - 1, n_grid))
    for k in range(n_grid - 1):
        D1[k, k] = -1
        D1[k, k + 1] = 1

    f = cp.Variable(n_grid)

    # Objective: weighted least squares + smoothness penalty
    objective = cp.sum(cp.multiply(weights, cp.square(A @ f - ys))) + alpha * cp.sum_squares(D2 @ f)

    # Constraint: monotone increasing/decreasing
    if increasing:
        constraints = [D1 @ f >= 0]
    else:
        constraints = [D1 @ f <= 0]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    return grid_x, f.value
