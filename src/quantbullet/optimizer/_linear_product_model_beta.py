import numpy as np
from scipy.optimize import least_squares

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
        # iterate in the same order used by jacobian/flatten/unflatten
        for key in self.block_names:
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

    # ---------- NEW (internal helpers for "one fixed per block") ----------
    def _block_sizes(self):
        return {k: len(self.feature_groups_[k]) for k in self.block_names}

    def _fixed_index(self, block_name):
        # Fix the FIRST coefficient in each block to 1
        return 0

    def _flatten_free(self, params_blocks):
        """Flatten only FREE params (skip fixed index per block)."""
        flat = []
        for k in self.block_names:
            idx0 = self._fixed_index(k)
            theta = params_blocks[k]
            if idx0 == 0:
                flat.extend(theta[1:])
            else:
                flat.extend(np.r_[theta[:idx0], theta[idx0+1:]])
        return np.array(flat, dtype=float)

    def _unflatten_free_to_full(self, flat_free):
        """Unflatten FREE params into FULL blocks with fixed=1 inserted."""
        sizes = self._block_sizes()
        params_blocks = {}
        pos = 0
        for k in self.block_names:
            p = sizes[k]
            idx0 = self._fixed_index(k)
            # gather free part for this block
            free_len = p - 1
            free_vec = flat_free[pos: pos + free_len]
            pos += free_len
            # rebuild full theta with fixed 1 at idx0
            if idx0 == 0:
                theta = np.concatenate(([1.0], free_vec))
            elif idx0 == p - 1:
                theta = np.concatenate((free_vec, [1.0]))
            else:
                theta = np.concatenate((free_vec[:idx0], [1.0], free_vec[idx0:]))
            params_blocks[k] = theta
        return params_blocks

    def jacobian(self, params_blocks, X_blocks):
        """
        Jacobian with respect to the FREE parameters only (fixed #0 per block is excluded).
        """
        n_obs = X_blocks[self.block_names[0]].shape[0]
        # prefix/suffix products
        inners = []
        for key in self.block_names:
            theta = params_blocks[key]
            X = X_blocks[key]
            inners.append(X @ theta)

        prefix = [np.ones(n_obs)]
        for inner in inners[:-1]:
            prefix.append(prefix[-1] * inner)
        suffix = [np.ones(n_obs)]
        for inner in inners[:0:-1]:
            suffix.append(suffix[-1] * inner)
        suffix = suffix[::-1]

        # total free columns = sum_k (p_k - 1)
        total_free = sum(X_blocks[k].shape[1] - 1 for k in self.block_names)
        J = np.zeros((n_obs, total_free), dtype=float)

        col = 0
        for j, key in enumerate(self.block_names):
            Xk = X_blocks[key]
            idx0 = self._fixed_index(key)  # fixed column index (0)
            prod_other = prefix[j] * suffix[j]
            # take all columns EXCEPT the fixed one
            if idx0 == 0:
                Xk_free = Xk[:, 1:]
            else:
                Xk_free = np.concatenate([Xk[:, :idx0], Xk[:, idx0+1:]], axis=1)
            J[:, col: col + Xk_free.shape[1]] = Xk_free * prod_other[:, None]
            col += Xk_free.shape[1]
        return J
    
    # ---------- Original flatten/unflatten (FULL params) ----------
    def flatten_params(self, params_blocks):
        """Flatten FULL parameters (includes fixed entries)."""
        if not isinstance(params_blocks, dict):
            raise ValueError("params_blocks must be a dictionary.")
        flat_params = []
        for key in self.block_names:
            if key not in params_blocks:
                raise ValueError(f"Feature block '{key}' not found in params_blocks.")
            flat_params.extend(params_blocks[key])
        return np.array(flat_params, dtype=float)
    
    def unflatten_params(self, flat_params):
        """Unflatten FULL parameter vector (includes fixed entries)."""
        if not isinstance(flat_params, np.ndarray):
            raise ValueError("flat_params must be a numpy array.")
        params_blocks = {}
        start = 0
        for key in self.block_names:
            n_features = len(self.feature_groups_[key])
            params_blocks[key] = flat_params[start: start + n_features]
            start += n_features
        return params_blocks

    # ---------- Fit using FREE parameters, store FULL coefficients ----------
    def fit(self, X, y, feature_groups, use_jacobian=True, **kwargs):
        self.feature_groups_ = feature_groups
        _ = self.n_features  # touch to ensure set/compute
        
        y = np.asarray(y, dtype=float).ravel()
        X_blocks = { key: X[feature_groups[key]].values for key in feature_groups }

        # init FULL thetas (all ones), then flatten FREE to pass to optimizer
        init_params_blocks_full = { key: np.ones(len(feature_groups[key]), dtype=float)
                                    for key in feature_groups }
        init_free = self._flatten_free(init_params_blocks_full)

        def residuals(free_params):
            params_blocks_full = self._unflatten_free_to_full(free_params)
            return self.forward(params_blocks_full, X_blocks) - y

        def call_jacobian(free_params):
            params_blocks_full = self._unflatten_free_to_full(free_params)
            return self.jacobian(params_blocks_full, X_blocks)

        if use_jacobian:
            kwargs["jac"] = call_jacobian

        result = least_squares(
            residuals,
            xtol=self.xtol,
            ftol=self.ftol,
            gtol=self.gtol,
            x0=init_free,
            **kwargs
        )

        # Rebuild FULL parameter vector and store for predict()/coef_dict
        coef_blocks_full = self._unflatten_free_to_full(result.x)
        self.coef_ = self.flatten_params(coef_blocks_full)  # full vector (includes fixed 1s)
        self.result_ = result
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model not fitted yet.")
        coef_blocks = self.unflatten_params(self.coef_)  # FULL blocks (with fixed 1s)
        X_blocks = { key: X[self.feature_groups_[key]].values for key in self.feature_groups_ }
        return self.forward(coef_blocks, X_blocks)

    @property
    def coef_dict(self):
        """Return coefficients grouped by feature group (FULL, including fixed entries)."""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet.")
        if self.feature_groups_ is None:
            raise ValueError("coef_dict only available when model was fit with DataFrame + feature_groups.")
        coef_blocks = self.unflatten_params(self.coef_)
        out = {}
        for group, cols in self.feature_groups_.items():
            out[group] = dict(zip(cols, coef_blocks[group]))
        return out
