import numpy as np
import pandas as pd
from scipy.optimize import least_squares


class LinearProductRegressionModelOLS:
    def __init__(self):
        self.feature_groups_ = None
        self.mse_history_ = []
        self.params_history_ = []
        self.coef_ = None
        self.block_means_ = {}

    def _clear_history(self):
        """Clear the history of MSE and parameters."""
        self.mse_history_ = []
        self.params_history_ = []
        self.coef_ = None
        self.block_means_ = {}

    def fit( self, X, y, feature_groups, early_stopping_rounds=None, n_iterations=10, verbose=1 ):
        self.feature_groups_ = feature_groups
        feature_data_blocks = { key: X[feature_groups[key]].values for key in feature_groups }
        params_blocks = {key: np.ones(len(feature_groups[key]), dtype=float) for key in feature_groups}
        
        self._clear_history()
        for i in range(n_iterations):
            for feature_group in feature_groups:
                floating_data = feature_data_blocks[feature_group]

                fixed_params_blocks = { key: params_blocks[key] for key in feature_groups if key != feature_group }
                fixed_data_blocks = { key: feature_data_blocks[key] for key in feature_groups if key != feature_group }
                fixed_predictions = self.forward(fixed_params_blocks, fixed_data_blocks)
                residuals = y / fixed_predictions

                # fit a OLS model to the residuals using matrix operations
                floating_params = np.linalg.lstsq(floating_data, residuals, rcond=None)[0]
                params_blocks[feature_group] = floating_params
                
            predictions = self.forward(params_blocks, feature_data_blocks)
            mse = np.mean((y - predictions) ** 2)
            self.mse_history_.append(mse)
            self.params_history_.append(params_blocks)
            
            if verbose > 0:
                print(f"Iteration {i+1}/{n_iterations}, MSE: {mse:.4f}")
            
            # add the early stopping condition
            if early_stopping_rounds is not None and len(self.mse_history_) > early_stopping_rounds:
                if self.mse_history_[-1] >= self.mse_history_[-early_stopping_rounds]:
                    print(f"Early stopping at iteration {i+1} with MSE: {mse:.4f}")
                    self.coef_ = self.params_history_[-early_stopping_rounds]
                    break
                
        if self.coef_ is None:
            self.coef_ = self.params_history_[-1]
        
        # archive the mean of each block's predictions
        for key in feature_groups:
            block_params = self.coef_[key]
            block_data = feature_data_blocks[key]
            block_pred = self.forward({key: block_params}, {key: block_data})
            block_mean = np.mean(block_pred)
            self.block_means_[key] = block_mean
            
        return self
    
    def predict( self, X ):
        if self.feature_groups_ is None or self.coef_ is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")
        feature_data_blocks = { key: X[self.feature_groups_[key]].values for key in self.feature_groups_ }
        return self.forward(self.coef_, feature_data_blocks)
    
    @property
    def coef_dict(self):
        """Return coefficients grouped by feature group."""
        return self._coef_to_coef_dict(self.coef_)
    
    @property
    def bias_one_coef_dict(self):
        """Return bias-one normalized coefficients grouped by feature group."""
        return self._coef_to_coef_dict(self.bias_one_coef_)
    
    @property
    def normalized_coef_dict(self):
        """Return block-mean normalized coefficients grouped by feature group."""
        return self._coef_to_coef_dict(self.normalized_coef_)
    
    def _coef_to_coef_dict(self, coef):
        """Convert coef_ dict to a nested dictionary with feature names."""
        if self.feature_groups_ is None:
            raise ValueError("feature_groups_ is not set.")
        coef_dict = {}
        for group, features in self.feature_groups_.items():
            coef_dict[group] = {features[i]: coef[group][i] for i in range(len(features))}
        return coef_dict
    
    @property
    def bias_one_coef_(self):
        # assume the first feature in each group is the bias term
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")
        bias_one_coef = {}
        for group, coef in self.coef_.items():
            bias_coef = coef[0]
            bias_one_coef[group] = coef / bias_coef
        return bias_one_coef
    
    @property
    def normalized_coef_(self):
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")
        normalized_coef = {}
        for group, coef in self.coef_.items():
            block_mean = self.block_means_.get(group)
            normalized_coef[group] = coef / block_mean
        return normalized_coef
    
    def forward(self, params_blocks, X_blocks):
        """
        Compute the forward pass for the model.

        Parameters
        ----------
        params_blocks : dict
            A dictionary mapping feature block names to their parameter vectors.
        X_blocks : dict
            A dictionary mapping feature block names to their input data matrices.

        Returns
        -------
        np.ndarray
            The model's predictions for the input data.
        """
        # Find any block to get n_obs
        for key in X_blocks:
            n_obs = X_blocks[key].shape[0]
            break
        else:
            raise ValueError("X_blocks is empty.")

        result = np.ones(n_obs, dtype=float)
        for key in params_blocks:
            if key not in X_blocks:
                raise ValueError(f"Feature block '{key}' not found in input blocks.")
            result *= np.dot(X_blocks[key], params_blocks[key])

        return result

class LinearProductRegressionModelScipy:
    def __init__(self, xtol=1e-8, ftol=1e-8, gtol=1e-8):
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
        y_hat = np.ones(n_obs)
        for theta, X in self._split_params(params, X_blocks):
            inner = X @ theta
            y_hat = y_hat * inner
        return y_hat

    def jacobian(self, params, X_blocks):
        n_obs = X_blocks[0].shape[0]
        n_features = sum(X.shape[1] for X in X_blocks)
        J = np.zeros((n_obs, n_features))

        # prefix/suffix products to avoid recomputing
        inners = []
        for theta, X in self._split_params(params, X_blocks):
            inner = X @ theta
            inners.append(inner)

        prefix = [np.ones(n_obs)]
        for inner in inners[:-1]:
            prefix.append(prefix[-1] * inner)
        suffix = [np.ones(n_obs)]
        for inner in inners[:0:-1]:
            suffix.append(suffix[-1] * inner)
        suffix = suffix[::-1]

        idx = 0
        for j, (theta, X) in enumerate(self._split_params(params, X_blocks)):
            prod_other = prefix[j] * suffix[j]
            J[:, idx: idx+X.shape[1]] = X * prod_other[:, None]
            idx += X.shape[1]
        return J

    # -----------------------------
    # Public API
    # -----------------------------
    def fit(self, X, y, feature_groups=None, init_params=None, use_jacobian=True, **kwargs):
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
            init_params = np.ones(n_features, dtype=float)
        else:
            init_params = np.asarray(init_params, dtype=float)
        
        y = np.asarray(y, dtype=float).ravel()

        def residuals(params):
            return self.forward(params, X_blocks) - y

        if use_jacobian:
            kwargs["jac"] = lambda p: self.jacobian(p, X_blocks)
        

        result = least_squares(
            residuals,
            xtol=self.xtol,
            ftol=self.ftol,
            gtol=self.gtol,
            x0=init_params,
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
