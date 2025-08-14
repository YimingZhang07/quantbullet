import numpy as np
import copy
import pandas as pd
from scipy.optimize import least_squares


class LinearProductModelOLS:
    def __init__(self):
        self.feature_groups_ = None
        self._clear_history()

    def _clear_history(self):
        """Clear the history of MSE and parameters."""
        self.mse_history_ = []
        self.params_history_ = []
        self.coef_ = None
        self.block_means_ = {}
        self.best_mse_ = float('inf')
        self.best_params_ = None
        self.global_scale_ = 1.0  # global scale for the model
        self.global_scale_history_ = []
        self.best_iteration_ = None

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

                # normalize the floating parameters
                # first calculate the mean of the floating predictions
                floating_predictions = floating_data @ floating_params
                floating_mean = np.mean(floating_predictions)
                
                if not np.isclose(floating_mean, 0):
                    # normalize the parameters by the mean
                    floating_params /= floating_mean
                    # update the global scale
                    self.global_scale_ = self.global_scale_ * floating_mean
                
                params_blocks[feature_group] = floating_params
              
            # track the training progress  
            predictions = self.forward(params_blocks, feature_data_blocks)
            mse = np.mean((y - predictions) ** 2)
            self.mse_history_.append(mse)
            self.params_history_.append(  copy.deepcopy(params_blocks) )
            self.global_scale_history_.append(self.global_scale_)
            
            # track the best parameters
            if mse < self.best_mse_:
                self.best_mse_ = mse
                self.best_params_ = copy.deepcopy(params_blocks)
                self.best_iteration_ = i
            
            if verbose > 0:
                print(f"Iteration {i+1}/{n_iterations}, MSE: {mse:.4f}")
            
            # add the early stopping condition
            if early_stopping_rounds is not None and len(self.mse_history_) > early_stopping_rounds:
                if self.mse_history_[-1] >= self.mse_history_[-early_stopping_rounds]:
                    print(f"Early stopping at iteration {i+1} with MSE: {mse:.4f}")
                    break
                
        self.coef_ = copy.deepcopy(self.best_params_)
        self.global_scale_ = self.global_scale_history_[self.best_iteration_]
        
        # archive the mean of each block's predictions
        for key in feature_groups:
            block_params = self.coef_[key]
            block_data = feature_data_blocks[key]
            block_pred = self.forward({key: block_params}, {key: block_data}, ignore_global_scale=True)
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
    
    def forward(self, params_blocks, X_blocks, ignore_global_scale=False):
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

        if not ignore_global_scale:
            result = result * self.global_scale_
        return result

class LinearProductModelScipy:
    def __init__(self, xtol=1e-8, ftol=1e-8, gtol=1e-8):
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.coef_ = None
        self.feature_groups_ = None   # user-defined groups (dict[str, list[str]])

    # -----------------------------
    # Model math
    # -----------------------------
    def forward(self, params_blocks, X_blocks):
        for key in X_blocks:
            n_obs = X_blocks[key].shape[0]
            break
        else:
            raise ValueError("X_blocks is empty.")
        
        result = np.ones(n_obs, dtype=float)
        
        for key in params_blocks:
            if key not in X_blocks:
                raise ValueError(f"Feature block '{key}' not found in input blocks.")
            theta = params_blocks[key]
            X = X_blocks[key]
            inner = X @ theta
            result *= inner

        return result
    
    @property
    def n_features(self):
        if self.feature_groups_ is None:
            raise ValueError("feature_groups_ is not set.")
        return sum(len(cols) for cols in self.feature_groups_.values())
    
    @property
    def block_names(self):
        if self.feature_groups_ is None:
            raise ValueError("feature_groups_ is not set.")
        return list(self.feature_groups_.keys())

    def jacobian(self, params_blocks, X_blocks):
        n_obs = X_blocks[ self.block_names[0] ].shape[0]
        n_features = self.n_features
        
        J = np.zeros((n_obs, n_features), dtype=float)

        # prefix/suffix products to avoid recomputing
        inners = []
        for key in self.block_names:
            theta = params_blocks[key]
            X = X_blocks[key]
            inner = X @ theta
            inners.append(inner)

        prefix = [np.ones(n_obs)]
        for inner in inners[:-1]:
            prefix.append(prefix[-1] * inner)
        suffix = [np.ones(n_obs)]
        for inner in inners[:0:-1]:
            suffix.append(suffix[-1] * inner)
        suffix = suffix[::-1]
        
        # Fill the Jacobian
        col = 0
        for j, key in enumerate(self.block_names):
            X = X_blocks[key]
            prod_other = prefix[j] * suffix[j]
            J[:, col: col+X.shape[1]] = X * prod_other[:, None]
            col += X.shape[1]
            
        return J
    
    def flatten_params(self, params_blocks):
        """
        Flatten the parameters from a dictionary of blocks into a single array.
        """
        if not isinstance(params_blocks, dict):
            raise ValueError("params_blocks must be a dictionary.")
        
        flat_params = []
        for key in self.block_names:
            if key not in params_blocks:
                raise ValueError(f"Feature block '{key}' not found in params_blocks.")
            flat_params.extend(params_blocks[key])
        
        return np.array(flat_params, dtype=float)
    
    def unflatten_params(self, flat_params):
        """
        Unflatten the parameters from a single array into a dictionary of blocks.
        """
        if not isinstance(flat_params, np.ndarray):
            raise ValueError("flat_params must be a numpy array.")
        
        params_blocks = {}
        start = 0
        for key in self.block_names:
            if key not in self.feature_groups_:
                raise ValueError(f"Feature block '{key}' not found in feature_groups_.")
            n_features = len(self.feature_groups_[key])
            params_blocks[key] = flat_params[start: start + n_features]
            start += n_features
        
        return params_blocks

    def fit(self, X, y, feature_groups, use_jacobian=True, **kwargs):
        """
        Fit the model.
        """
        self.feature_groups_ = feature_groups
        n_features = self.n_features
        
        y = np.asarray(y, dtype=float).ravel()
        
        X_blocks = { key: X[feature_groups[key]].values for key in feature_groups }
        init_params_blocks = { key: np.ones(len(feature_groups[key]), dtype=float) for key in feature_groups }
        init_params = self.flatten_params(init_params_blocks)

        def residuals(params):
            params_blocks = self.unflatten_params(params)
            return self.forward(params_blocks, X_blocks) - y

        def call_jacobian(params):
            params_blocks = self.unflatten_params(params)
            return self.jacobian(params_blocks, X_blocks)

        if use_jacobian:
            kwargs["jac"] = call_jacobian

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

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model not fitted yet.")
        
        coef_blocks = self.unflatten_params(self.coef_)
        X_blocks = { key: X[self.feature_groups_[key]].values for key in self.feature_groups_ }
        
        return self.forward(coef_blocks, X_blocks)

    @property
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
