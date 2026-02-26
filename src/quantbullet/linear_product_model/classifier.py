import numpy as np
import pandas as pd
import copy
from scipy.optimize import minimize
from .utils import (
    minimize_clipped_cross_entropy_loss,
    log_loss,
    fit_logistic_no_intercept,
)

from .base import LinearProductClassifierBase, LinearProductModelBCD, memorize_fit_args

class LinearProductClassifierScipy(LinearProductClassifierBase):
    def __init__(self, gtol=1e-8, ftol=1e-8, eps=1e-3):
        # initialize the base class
        super().__init__()
        self.gtol = gtol
        self.ftol = ftol
        self.eps = eps
        self.coef_ = None

    # -----------------------------
    # Model math
    # -----------------------------
    def forward_raw(self, params_blocks, X_blocks):
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
    
    def forward(self, params_blocks, X_blocks, ignore_global_scale=False):
        y_hat = self.forward_raw(params_blocks, X_blocks)
        return np.clip(y_hat, self.eps, 1 - self.eps)
    
    def objective(self, params, X_blocks, y):
        params_blocks = self.unflatten_params(params)
        y_hat = self.forward(params_blocks, X_blocks)
        cross_entropy = log_loss(y_hat, y) * len(y)
        return cross_entropy

    def jacobian(self, params, X_blocks, y):
        """
        Analytic gradient of the cross-entropy under your current parameterization:
        r = product_k (X_k @ theta_k), y_hat = clip(r, eps, 1-eps).
        For clipped points, gradient = 0.
        """
        params = np.asarray(params, dtype=float)
        params_blocks = self.unflatten_params(params)

        # compute inner products per block and the raw product r
        inners = {}
        r = np.ones_like(y, dtype=float)
        for key in self.block_names:
            Xk = X_blocks[key]
            th = params_blocks[key]
            ak = Xk @ th                # shape (n,)
            inners[key] = ak
            r *= ak

        eps = self.eps
        y_hat = np.clip(r, eps, 1.0 - eps)

        # mask for interior points (not clipped)
        mask = (r > eps) & (r < 1.0 - eps)

        # if everything is clipped, gradient is zero
        if not np.any(mask):
            grad_blocks = {key: np.zeros_like(params_blocks[key]) for key in self.block_names}
            return self.flatten_params(grad_blocks)

        # dL/dy_hat = (y_hat - y) / (y_hat * (1 - y_hat)); dy_hat/dr = 1 in interior
        slope = np.zeros_like(y_hat)
        slope[mask] = (y_hat[mask] - y[mask]) / (y_hat[mask] * (1.0 - y_hat[mask]))

        # per-block gradient: X_k^T [ slope * (r / inner_k) ] on interior
        tiny = 1e-12
        grad_blocks = {}
        for key in self.block_names:
            Xk = X_blocks[key]
            ak = inners[key]
            denom = np.where(np.abs(ak) < tiny, np.sign(ak) * tiny + tiny, ak)

            factor = np.zeros_like(r)
            factor[mask] = r[mask] / denom[mask]   # only interior contributes

            w = slope * factor                     # sample-wise weights
            grad_blocks[key] = Xk.T @ w

        return self.flatten_params(grad_blocks)

    def fit(self, X, y, feature_groups, init_params=None, use_jacobian=True):
        """
        Fit the model.
        """
        self.feature_groups_ = feature_groups
        
        y = np.asarray(y, dtype=float).ravel()
        X_blocks = { key: X[feature_groups[key]].values for key in feature_groups }
        
        init_params, _ = self.infer_init_params(init_params, X_blocks, y)
        
        def cb(params):
            loss = self.objective(params, X_blocks, y)
            print( f"Iter {cb.iter_count}: {loss}" )
            cb.iter_count += 1

        cb.iter_count = 1
        
        if use_jacobian:
            jac = lambda p: self.jacobian(p, X_blocks, y)
        else:
            jac = None

        result = minimize(
            fun=lambda p: self.objective(p, X_blocks, y),
            jac=jac,
            x0=init_params,
            method='L-BFGS-B',
            callback=cb,
            options={'gtol': self.gtol, 'ftol': self.ftol}
        )
        
        self.coef_ = self.unflatten_params(result.x)
        self.result_ = result
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model not fitted yet.")
        
        X_blocks = { key: X[self.feature_groups_[key]].values for key in self.feature_groups_ }

        return self.forward(self.coef_, X_blocks)

class LinearProductClassifierBCD( LinearProductClassifierBase, LinearProductModelBCD ):
    def __init__(self, eps=1e-3):
        LinearProductClassifierBase.__init__(self)
        LinearProductModelBCD.__init__(self)
        self.eps = eps

    def loss_function(self, y_hat, y):
        return log_loss(y_hat, y)

    @memorize_fit_args
    def fit( self, X, y, feature_groups, init_params=None, early_stopping_rounds=None, n_iterations=10, verbose=1 ):
        self._reset_history()
        self.feature_groups_ = feature_groups
        data_blocks = { key: X[feature_groups[key]].values for key in feature_groups }
        
        if init_params is None:
            self.global_scalar_ = np.mean(y)
            _, params_blocks = self.infer_init_params(init_params, data_blocks, np.ones_like(y))
        else:
            _, params_blocks = self.infer_init_params(init_params, data_blocks, y)


        for i in range(n_iterations):
            for feature_group in feature_groups:
                floating_data = data_blocks[feature_group]

                fixed_params_blocks = { key: params_blocks[key] for key in feature_groups if key != feature_group }
                fixed_data_blocks = { key: data_blocks[key] for key in feature_groups if key != feature_group }

                if len(fixed_params_blocks) == 0:
                    fixed_predictions = np.ones(floating_data.shape[0], dtype=float)
                else:
                    fixed_predictions = self.forward(fixed_params_blocks, fixed_data_blocks, ignore_global_scale=True)

                # treat the products of other blocks as a vector to scale the X
                # then just minimize the clipped cross-entropy loss on this block
                floating_data = floating_data * fixed_predictions[:, None]
                floating_params, status = minimize_clipped_cross_entropy_loss( self.global_scalar_ * floating_data, y, beta0=None, eps=self.eps )
                
                if not status.success:
                    print(f"Warning: Optimization for block '{feature_group}' did not converge: {status.message}")

                # normalize the floating parameters
                floating_predictions = floating_data @ floating_params
                floating_mean = np.mean(floating_predictions)
                
                if not np.isclose(floating_mean, 0):
                    floating_params /= floating_mean
                    self.global_scalar_ = self.global_scalar_ * floating_mean
                else:
                    print(f"Warning: Mean of predictions for block '{feature_group}' is close to zero. Skipping normalization.")
                
                params_blocks[feature_group] = floating_params
              
            # track the training progress  
            predictions = self.forward(params_blocks, data_blocks)
            loss = self.loss_function(predictions, y)
            self.loss_history_.append(loss)
            self.coef_history_.append( copy.deepcopy(params_blocks) )
            self.global_scalar_history_.append(self.global_scalar_)
            
            # track the best parameters
            if loss < self.best_loss_:
                self.best_loss_ = loss
                self.best_params_ = copy.deepcopy(params_blocks)
                self.best_iteration_ = i
            
            if verbose > 0:
                print(f"Iteration {i+1}/{n_iterations}, Loss: {loss:.4f}")
            
            # add the early stopping condition
            if early_stopping_rounds is not None and len(self.loss_history_) > early_stopping_rounds:
                if self.loss_history_[-1] >= self.loss_history_[-early_stopping_rounds]:
                    print(f"Early stopping at iteration {i+1} with Loss: {loss:.4f}")
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