import numpy as np
import copy
import pandas as pd
from scipy.optimize import least_squares
from .base import LinearProductModelBase, LinearProductModelBCD, LinearProductRegressorBase


class LinearProductRegressorBCD( LinearProductRegressorBase, LinearProductModelBCD ):
    def __init__(self):
        LinearProductModelBase.__init__(self)
        LinearProductModelBCD.__init__(self)

    def loss_function(self, y_hat, y):
        return np.mean((y - y_hat) ** 2)

    def fit( self, X, y, feature_groups, init_params=None, early_stopping_rounds=5, n_iterations=20, verbose=1, cache_qr_decomp=False, ftol=1e-5 ):
        self._reset_history( cache_qr_decomp=cache_qr_decomp )
        self.feature_groups_ = feature_groups
        data_blocks = { key: X[feature_groups[key]].values for key in feature_groups }

        if init_params is None:
            # absorb the mean of y to the global scaler and then init the block params so that they give a prediction of 1.
            # the function infer_init_params will simply use the mean of y here, so a constant vector is good.
            self.global_scalar_ = np.mean(y)
            _, params_blocks = self.infer_init_params(init_params, data_blocks, np.ones_like(y))
        else:
            _, params_blocks = self.infer_init_params(init_params, data_blocks, y)

        for i in range(n_iterations):
            for feature_group in feature_groups:
                floating_data = data_blocks[feature_group]
                fixed_params_blocks = { key: params_blocks[key] for key in feature_groups if key != feature_group }
                fixed_data_blocks = { key: data_blocks[key] for key in feature_groups if key != feature_group }
                # We hope to maintain the average output of each feature group is 1
                # so the global scaler is not used to scale the floating data util the actual regression step
                fixed_predictions = self.forward(fixed_params_blocks, fixed_data_blocks, ignore_global_scale=True)

                if not cache_qr_decomp:
                    floating_data = floating_data * fixed_predictions[:, None]
                    floating_params = np.linalg.lstsq( self.global_scalar_ * floating_data, y, rcond=None)[0]
                else:
                    # use the cached QR decomposition to solve the least squares problem so that we do not need to recompute the inverse of X'X every time
                    # the downside is we need to put the global scaler and fixed predictions into the y cause they will change every iteration
                    if feature_group not in self.qr_decomp_cache_:
                        Q, R = np.linalg.qr( floating_data )
                        self.qr_decomp_cache_[feature_group] = (Q, R)
                    else:
                        Q, R = self.qr_decomp_cache_[feature_group]
                    scaled_y = y / self.global_scalar_ / fixed_predictions
                    floating_params = np.linalg.solve( R, Q.T @ scaled_y )

                # normalize the floating parameters by its mean so that each block's prediction has a mean of 1
                floating_predictions = floating_data @ floating_params
                floating_mean = np.mean(floating_predictions)
                
                if not np.isclose(floating_mean, 0):
                    floating_params /= floating_mean
                    self.global_scalar_ = self.global_scalar_ * floating_mean
                else:
                    print(f"Warning: floating mean is close to zero for feature group {feature_group} at iteration {i}. Skipping normalization.")
                
                params_blocks[feature_group] = floating_params
              
            # track the training progress  
            predictions = self.forward(params_blocks, data_blocks)
            loss = self.loss_function(predictions, y)
            self.loss_history_.append(loss)
            self.params_history_.append(  copy.deepcopy(params_blocks) )
            self.global_scalar_history_.append(self.global_scalar_)
            
            # track the best parameters
            if loss < self.best_loss_:
                self.best_loss_ = loss
                self.best_params_ = copy.deepcopy(params_blocks)
                self.best_iteration_ = i
            
            if verbose > 0:
                print(f"Iteration {i+1}/{n_iterations}, Loss: {loss:.4e}")
            
            # add the early stopping condition
            if early_stopping_rounds is not None and len(self.loss_history_) > early_stopping_rounds:
                if self.loss_history_[-1] >= self.loss_history_[-early_stopping_rounds]:
                    print(f"Early stopping at iteration {i+1} with Loss: {loss:.4e}")
                    break
                
            if ftol is not None and len(self.loss_history_) >= 2:
                if abs(self.loss_history_[-1] / self.loss_history_[-2]) > 1 - ftol:
                    print(f"Converged at iteration {i+1} with Loss: {loss:.4e}")
                    break
                
        self.coef_ = copy.deepcopy(self.best_params_)
        self.global_scalar_ = self.global_scalar_history_[self.best_iteration_]
        
        # archive the mean of each block's predictions
        for key in feature_groups:
            block_params = self.coef_[key]
            block_data = data_blocks[key]
            block_pred = self.forward({key: block_params}, {key: block_data}, ignore_global_scale=True)
            block_mean = np.mean(block_pred)
            self.block_means_[key] = block_mean
            
        return self
    
    def predict( self, X ):
        if self.feature_groups_ is None or self.coef_ is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")
        data_blocks = { key: X[self.feature_groups_[key]].values for key in self.feature_groups_ }
        return self.forward(self.coef_, data_blocks)
    
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
            result = result * self.global_scalar_
        return result


class LinearProductRegressorScipy(LinearProductRegressorBase):
    def __init__(self, xtol=1e-8, ftol=1e-8, gtol=1e-8):
        # initialize the base class
        super().__init__()
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.coef_ = None

    # -----------------------------
    # Model math
    # -----------------------------
    def forward(self, params_blocks, X_blocks, ignore_global_scale=False):
        for key in X_blocks:
            n_obs = X_blocks[key].shape[0]
            break
        else:
            raise ValueError("X_blocks is empty.")
        
        result = np.ones(n_obs, dtype=float)

        for key in params_blocks:
            theta = params_blocks[key]
            X = X_blocks[key]
            inner = X @ theta
            result *= inner

        return result

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

    def fit(self, X, y, feature_groups, init_params=1, use_jacobian=True, **kwargs):
        """
        Fit the model.
        """
        self.feature_groups_ = feature_groups
        
        y = np.asarray(y, dtype=float).ravel()
        X_blocks = { key: X[feature_groups[key]].values for key in feature_groups }
        init_params, _ = self.infer_init_params(init_params, X_blocks, y)

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
        self.coef_ = self.unflatten_params(result.x)
        self.result_ = result
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model not fitted yet.")
        
        X_blocks = { key: X[self.feature_groups_[key]].values for key in self.feature_groups_ }
        
        return self.forward(self.coef_, X_blocks)
