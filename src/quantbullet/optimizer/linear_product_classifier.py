import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .linear_product_shared import init_betas_by_response_mean

class LinearProductClassifierScipy:
    def __init__(self, gtol=1e-8, ftol=1e-8, eps=1e-3):
        self.gtol = gtol
        self.ftol = ftol
        self.eps = eps
        self.coef_ = None
        self.feature_groups_ = None   # user-defined groups (dict[str, list[str]])

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
        
        if init_params is None:
            # we cannot use 1s as initial parameters anymore, as this leads to >1 predicted values and clipped to 1 for all observations
            # making it impossible to optimize;
            # Therefore we initialize a constant value that on average predicts the true probability
            true_prob = np.mean(y)
            n_blocks = len(self.block_names)
            block_target = true_prob ** (1 / n_blocks)
            init_params_blocks = { key: init_betas_by_response_mean(X_blocks[key], block_target) for key in self.block_names }
            print(f"Using initial params: {init_params_blocks}")
            init_params = self.flatten_params(init_params_blocks)
        else:
            if np.isscalar(init_params):
                init_params_blocks = { key: np.full(len(feature_groups[key]), float(init_params), dtype=float) for key in self.block_names }
                init_params = self.flatten_params(init_params_blocks)
            elif isinstance(init_params, np.ndarray):
                if len(init_params) != self.n_features:
                    raise ValueError(f"init_params length {len(init_params)} does not match number of features {self.n_features}.")
                else:
                    init_params = np.asarray(init_params, dtype=float)
                    init_params_blocks = self.unflatten_params(init_params)
            else:
                raise ValueError("init_params must be None, a numpy array, or a scalar.")
        
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
