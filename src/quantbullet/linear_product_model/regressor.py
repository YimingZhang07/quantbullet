import numpy as np
import pandas as pd
import time

from typing import Dict
from scipy.optimize import least_squares

from .base import LinearProductModelBCD, LinearProductRegressorBase, InteractionCoef, memorize_fit_args
from .datacontainer import ProductModelDataContainer
from .forms import resolve_form
from .solvers import BCDSolver


class LinearProductRegressorBCD( LinearProductRegressorBase, LinearProductModelBCD ):
    # Solver that ``fit`` hands the iteration to; a subclass may swap in a variant.
    bcd_solver = BCDSolver

    def __init__(self):
        LinearProductRegressorBase.__init__(self)
        LinearProductModelBCD.__init__(self)
        self.interactions_ = {}

    def loss_function(self, y_hat, y, weights=None):
        return self.form_.loss(y, y_hat, weights)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cat_series(coef, orig):
        """The ``by`` column an interaction group needs for ``block_factor``; ``None`` for other groups."""
        return orig[coef.by] if isinstance(coef, InteractionCoef) else None

    def _resolve_blocks_and_orig(self, X):
        """Dispatch X type → (orig_df, get_block_fn)."""
        if isinstance(X, ProductModelDataContainer):
            return X.orig, X.get_expanded_array_for_feature_group
        elif isinstance(X, pd.DataFrame):
            return X, lambda g: X[self.feature_groups_[g]].values
        raise ValueError("Invalid input type. Expected ProductModelDataContainer or pd.DataFrame.")

    def _compute_block_means(self, data_blocks, X, weights=None):
        """Recompute ``block_means_`` from final ``coef_`` (including A/E scalars for interactions).

        Dispatch goes through ``block_factor`` so that frozen submodel blocks
        report the submodel's own prediction.  Multiplying the raw block by the
        placeholder coefficient left in ``coef_`` would produce a meaningless mean.
        """
        for key in self.feature_groups_:
            coef = self.coef_[key]
            cat_series = X.orig[coef.by] if isinstance(coef, InteractionCoef) else None
            block_pred = self.block_factor(key, data_blocks[key], cat_series=cat_series)
            self.block_means_[key] = np.average(block_pred, weights=weights) if weights is not None else np.mean(block_pred)

    # ------------------------------------------------------------------
    # fit (public entry point)
    # ------------------------------------------------------------------

    @memorize_fit_args
    def fit( self, X: ProductModelDataContainer, feature_groups: Dict, submodels: Dict=None,
             interactions: Dict=None, interaction_submodels: Dict=None,
             init_params=None, early_stopping_rounds=5, n_iterations=20, force_rounds=5, verbose=1, ftol=1e-5,
             cache_qr_decomp=False, offset_y=None, use_svd=False, weights=None, loss: str = 'mse' ):

        fit_started = time.perf_counter()
        # ---- 1) setup model state ----
        resolve_form("identity", loss)  # rejects unsupported losses before the model is reset
        self.loss_ = loss
        self._reset_history( cache_qr_decomp=cache_qr_decomp )
        self.feature_groups_ = feature_groups
        self.submodels_ = submodels or {}
        self.interactions_ = interactions or {}
        self.fit_timing_ = {'setup_seconds': 0., 'block_seconds': {group: 0. for group in feature_groups},
                            'sweep_seconds': [], 'total_seconds': 0.}

        for parent, cat_var in self.interactions_.items():
            if cat_var not in feature_groups:
                raise ValueError(
                    f"Interaction by-variable '{cat_var}' must be included in feature_groups "
                    f"as a separate categorical block (required for '{parent}' interaction)."
                )

        # ---- 2) validate inputs and build canonical arrays ----
        data_blocks = X.get_expanded_array_dict( list( feature_groups.keys() ) )

        if weights is not None:
            weights = np.asarray(weights).ravel()
            if weights.ndim != 1 or weights.shape[0] != X.shape[0]:
                raise ValueError("Weights must be a 1D array with the same length as the number of observations.")

        if X.response is None:
            raise ValueError("Response variable is not provided in the data container.")
        y = X.response
        if offset_y is not None:
            self.offset_y = offset_y
            y = y + offset_y

        # ---- 3) iterate ----
        solver = self.bcd_solver(
            self, n_iterations=n_iterations, early_stopping_rounds=early_stopping_rounds, force_rounds=force_rounds,
            ftol=ftol, cache_qr_decomp=cache_qr_decomp, use_svd=use_svd, verbose=verbose)
        solver.run(X, data_blocks, y, weights, init_params, interaction_submodels or {}, started=fit_started)
        return self

    # ------------------------------------------------------------------
    # predict  (override to handle InteractionCoef in coef_)
    # ------------------------------------------------------------------

    def predict(self, X: ProductModelDataContainer | pd.DataFrame):
        if self.feature_groups_ is None or self.coef_ is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")

        if not self.interactions_:
            return super().predict(X)

        orig, get_block = self._resolve_blocks_and_orig(X)

        n_obs = X.shape[0]
        result = np.ones(n_obs, dtype=float) * self.global_scalar_

        for group in self.feature_groups_:
            coef = self.coef_.get(group)
            if coef is None:
                continue
            result *= self.block_factor(group, get_block(group), cat_series=self._cat_series(coef, orig))

        if self.offset_y is not None:
            result -= self.offset_y
        return result

    # ------------------------------------------------------------------
    # single / leave-out  (override to handle interactions)
    # ------------------------------------------------------------------

    def single_feature_group_predict(self, group_to_include, X, params_dict=None, ignore_global_scale=True):
        coef = self.coef_.get(group_to_include)
        if not isinstance(coef, InteractionCoef):
            return super().single_feature_group_predict(group_to_include, X, params_dict, ignore_global_scale)

        orig, get_block = self._resolve_blocks_and_orig(X)
        return self.block_factor(group_to_include, get_block(group_to_include), cat_series=orig[coef.by])

    def leave_out_feature_group_predict(self, group_to_exclude, X,
                                        params_dict=None, ignore_global_scale=False):
        if not self.interactions_:
            return super().leave_out_feature_group_predict(
                group_to_exclude, X, params_dict, ignore_global_scale)

        if group_to_exclude not in self.feature_groups_:
            raise ValueError(f"Feature group '{group_to_exclude}' not found in feature_groups_.")

        orig, get_block = self._resolve_blocks_and_orig(X)

        n_obs = X.shape[0]
        result = np.ones(n_obs, dtype=float)
        if not ignore_global_scale:
            result *= self.global_scalar_

        for group in self.feature_groups_:
            if group == group_to_exclude:
                continue
            coef = self.coef_.get(group)
            if coef is None:
                continue
            result *= self.block_factor(group, get_block(group), cat_series=self._cat_series(coef, orig))

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
