from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from scipy.optimize import minimize


class ConstrainedLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Linear regression with per-feature coefficient constraints and optional custom loss.

    Parameters:
    -----------
    coef_constraints : list of {"+" | "-" | None | (low, high)}, optional
        Constraint per feature coefficient (excluding intercept):
            "+"     → coef >= 0
            "-"     → coef <= 0
            None    → no constraint
            (a, b)  → bounded: a <= coef <= b
    fit_intercept : bool
        Whether to fit the intercept term.
    loss : callable
        Custom loss function. Must take (y_true, y_pred) and return a float.
    """

    def __init__(self, coef_constraints=None, fit_intercept=True, loss=None):
        self.coef_constraints = coef_constraints
        self.fit_intercept = fit_intercept
        self.loss = loss
        
    def get_loss_function(self):
        """
        Returns the custom loss function if set, otherwise returns None.
        """
        if self.loss is not None:
            return self.loss
        
        def default_loss(y_true, y_pred):
            """
            Default loss function: Mean Squared Error (MSE).
            """
            return np.mean((y_true - y_pred) ** 2)
        
        return default_loss

    def _build_bounds(self, n_features):
        if self.coef_constraints is None:
            return [(None, None)] * (n_features + int(self.fit_intercept))
        
        if len(self.coef_constraints) != n_features:
            raise ValueError("Length of coef_constraints must match number of features.")

        bounds = [(None, None)] if self.fit_intercept else []
        for c in self.coef_constraints:
            if c == "+":
                bounds.append((0, None))
            elif c == "-":
                bounds.append((None, 0))
            elif isinstance(c, tuple) and len(c) == 2:
                bounds.append(c)
            elif c is None:
                bounds.append((None, None))
            else:
                raise ValueError(f"Invalid constraint: {c}")
        return bounds

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # Design matrix
        if self.fit_intercept:
            X_design = np.hstack([np.ones((n_samples, 1)), X])
        else:
            X_design = X

        bounds = self._build_bounds(n_features)

        loss_fn = self.get_loss_function()

        def objective_fn(beta):
            return loss_fn(y, X_design @ beta)

        beta0 = np.zeros(X_design.shape[1])
        result = minimize(objective_fn, beta0, bounds=bounds)
        if not result.success:
            raise RuntimeError("Optimization failed: " + result.message)

        self.coef_ = result.x[1:] if self.fit_intercept else result.x
        self.intercept_ = result.x[0] if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
