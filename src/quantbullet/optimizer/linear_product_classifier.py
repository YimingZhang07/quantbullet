import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .linear_product_shared import init_betas_by_response_mean, LinearProductModelBase

class LinearProductClassifierScipy(LinearProductModelBase):
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
        
        for key in self.block_names:
            theta = params_blocks[key]
            X = X_blocks[key]
            inner = X @ theta
            result *= inner

        return result
    
    def forward(self, params_blocks, X_blocks):
        y_hat = self.forward_raw(params_blocks, X_blocks)
        return np.clip(y_hat, self.eps, 1 - self.eps)
    
    def objective(self, params, X_blocks, y):
        params_blocks = self.unflatten_params(params)
        y_hat = self.forward(params_blocks, X_blocks)
        cross_entropy = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
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
        """Return coefficients grouped by feature group."""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet.")
        if self.feature_groups_ is None:
            raise ValueError("coef_dict only available when model was fit with DataFrame + feature_groups.")

        coef_blocks = self.unflatten_params(self.coef_)
        out = {}
        for group, cols in self.feature_groups_.items():
            out[group] = dict(zip(cols, coef_blocks[group]))
        return out
