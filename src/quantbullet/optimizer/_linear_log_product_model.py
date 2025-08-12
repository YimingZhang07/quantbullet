import numpy as np
import pandas as pd
from scipy.optimize import least_squares


class PiecewiseLogProductModel:
    def __init__(self, min_log_value=1e-10, xtol=1e-8, ftol=1e-8, gtol=1e-8):
        self.min_log_value = min_log_value
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.coef_ = None
        self.feature_groups_ = None   # user-defined groups (dict[str, list[str]])
        self.feature_slices_ = None   # mapping group -> slice

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _split_params(self, params, X_blocks):
        idx = 0
        for X in X_blocks:
            n = X.shape[1]
            yield params[idx: idx+n], X
            idx += n

    def _build_blocks_from_df(self, X_df, feature_groups):
        """Convert DataFrame + feature groups dict -> X_blocks + slices."""
        X_blocks, slices = [], {}
        idx = 0
        for name, cols in feature_groups.items():
            block = X_df[cols].to_numpy()
            X_blocks.append(block)
            slices[name] = slice(idx, idx + block.shape[1])
            idx += block.shape[1]
        return X_blocks, slices

    # -----------------------------
    # Model math
    # -----------------------------
    def forward(self, params, X_blocks):
        n_obs = X_blocks[0].shape[0]
        y_hat = np.zeros(n_obs)
        for theta, X in self._split_params(params, X_blocks):
            inner = X @ theta
            inner = np.clip(inner, a_min=self.min_log_value, a_max=None)
            y_hat += np.log(inner)
        return y_hat

    def jacobian(self, params, X_blocks):
        n_obs = X_blocks[0].shape[0]
        n_features = sum(X.shape[1] for X in X_blocks)
        J = np.zeros((n_obs, n_features))

        idx = 0
        for theta, X in self._split_params(params, X_blocks):
            inner = X @ theta
            inner = np.clip(inner, a_min=self.min_log_value, a_max=None)
            J[:, idx: idx+X.shape[1]] = X / inner[:, None]
            idx += X.shape[1]
        return J

    # -----------------------------
    # Public API
    # -----------------------------
    def fit(self, X, y, feature_groups=None, init_params=None, **kwargs):
        """
        Fit the model.

        Parameters
        ----------
        X : pd.DataFrame or list of np.ndarray
            Either:
              - A DataFrame with columns covering feature_groups
              - A list of numpy arrays (blocks)
        y : array-like
        feature_groups : dict[str, list[str]], optional
            Required if X is a DataFrame.
        """
        if isinstance(X, pd.DataFrame):
            if feature_groups is None:
                raise ValueError("feature_groups must be provided when X is a DataFrame.")
            X_blocks, slices = self._build_blocks_from_df(X, feature_groups)
            self.feature_groups_ = feature_groups
            self.feature_slices_ = slices
        elif isinstance(X, list):
            X_blocks = X
            self.feature_groups_ = None
            self.feature_slices_ = None
        else:
            raise ValueError("X must be either a DataFrame or a list of arrays.")

        n_features = sum(X.shape[1] for X in X_blocks)
        if init_params is None:
            init_params = np.ones(n_features)

        def residuals(params):
            return self.forward(params, X_blocks) - y

        result = least_squares(
            residuals,
            x0=init_params,
            jac=lambda p: self.jacobian(p, X_blocks),
            xtol=self.xtol,
            ftol=self.ftol,
            gtol=self.gtol,
            **kwargs
        )
        self.coef_ = result.x
        self.result_ = result
        return self

    def predict(self, X, feature_groups=None):
        if self.coef_ is None:
            raise ValueError("Model not fitted yet.")

        if isinstance(X, pd.DataFrame):
            if feature_groups is None and self.feature_groups_ is None:
                raise ValueError("feature_groups must be provided for DataFrame input.")
            groups = feature_groups if feature_groups is not None else self.feature_groups_
            X_blocks, _ = self._build_blocks_from_df(X, groups)
        elif isinstance(X, list):
            X_blocks = X
        else:
            raise ValueError("X must be either a DataFrame or a list of arrays.")

        return self.forward(self.coef_, X_blocks)

    def coef_dict(self):
        """Return coefficients grouped by feature group (if trained with DataFrame)."""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet.")
        if self.feature_groups_ is None:
            raise ValueError("coef_dict only available when model was fit with DataFrame + feature_groups.")

        out = {}
        for group, cols in self.feature_groups_.items():
            sl = self.feature_slices_[group]
            out[group] = dict(zip(cols, self.coef_[sl]))
        return out
