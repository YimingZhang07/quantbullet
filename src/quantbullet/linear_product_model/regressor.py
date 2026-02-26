import copy
import numpy as np
import pandas as pd
import numexpr as ne

from typing import Dict, List, Optional, Union
from scipy.optimize import least_squares

from .base import LinearProductModelBCD, LinearProductRegressorBase, InteractionCoef, memorize_fit_args
from .utils import init_betas_by_response_mean
from .datacontainer import ProductModelDataContainer
from ._acceleration import ols_normal_equation, vector_product_numexpr_dict_values


class LinearProductRegressorBCD( LinearProductRegressorBase, LinearProductModelBCD ):
    def __init__(self):
        LinearProductRegressorBase.__init__(self)
        LinearProductModelBCD.__init__(self)
        self.interactions_ = {}

    def loss_function(self, y_hat, y):
        return np.mean((y - y_hat) ** 2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_interaction_block_pred(self, group, data_block, interaction_params, masks):
        """Assemble the combined prediction vector for an interaction group.

        Parameters
        ----------
        group : str
            Feature group name that has an interaction, e.g. ``'x1'``.
        data_block : np.ndarray, shape ``(n_obs, n_basis)``
            Expanded basis matrix for this group (e.g. FlatRampTransformer
            output).  Every row is one observation; columns are the knot-basis
            features.  All observations are present — masking by category
            happens inside this method.
        interaction_params : dict[str, dict[Any, np.ndarray | model]]
            Nested dict ``{group: {cat_val: coef}}``.
            ``coef`` is either an ``np.ndarray`` of shape ``(n_basis,)`` (OLS
            coefficients for that category) or a frozen submodel with a
            ``.predict()`` method.
        masks : dict[str, dict[Any, np.ndarray]]
            Nested dict ``{group: {cat_val: bool_mask}}``.
            ``bool_mask`` is a boolean array of shape ``(n_obs,)`` — True for
            observations belonging to that category.  The masks across
            categories are mutually exclusive and collectively exhaustive.

        Returns
        -------
        np.ndarray, shape ``(n_obs,)``
            Per-observation prediction for this block.  Observation *i* gets
            ``data_block[i] @ coef_c`` where *c* is its category.
        """
        combined = np.ones(data_block.shape[0], dtype=float)
        for cat_val, cat_coef in interaction_params[group].items():
            m = masks[group][cat_val]
            if hasattr(cat_coef, 'predict'):
                combined[m] = cat_coef.predict(data_block[m])
            else:
                combined[m] = data_block[m] @ cat_coef
        return combined

    def _absorb_block_mean(self, block_pred, weights=None):
        """Compute (weighted) mean of *block_pred*; if non-zero, absorb into ``global_scalar_``.

        Returns the mean so the caller can decide whether to divide params.
        """
        mean = np.average(block_pred, weights=weights) if weights is not None else np.mean(block_pred)
        if not np.isclose(mean, 0):
            self.global_scalar_ *= mean
        return mean

    def _resolve_blocks_and_orig(self, X):
        """Dispatch X type → (orig_df, get_block_fn)."""
        if isinstance(X, ProductModelDataContainer):
            return X.orig, X.get_expanded_array_for_feature_group
        elif isinstance(X, pd.DataFrame):
            return X, lambda g: X[self.feature_groups_[g]].values
        raise ValueError("Invalid input type. Expected ProductModelDataContainer or pd.DataFrame.")

    # ------------------------------------------------------------------
    # fit — initialisation helpers
    # ------------------------------------------------------------------

    def _init_interactions(self, X, data_blocks, feature_groups, interaction_submodels):
        """Build ``interaction_masks`` and ``interaction_params`` from raw data."""
        interaction_masks = {}
        interaction_params = {}
        for parent, cat_var in self.interactions_.items():
            if parent not in feature_groups:
                raise ValueError(f"Interaction parent '{parent}' not found in feature_groups.")
            categories = sorted(X.orig[cat_var].dropna().unique(), key=str)
            parent_subs = interaction_submodels.get(parent, {})
            interaction_masks[parent] = {}
            interaction_params[parent] = {}
            for cat_val in categories:
                interaction_masks[parent][cat_val] = (X.orig[cat_var] == cat_val).values
                if cat_val in parent_subs:
                    interaction_params[parent][cat_val] = parent_subs[cat_val]
                else:
                    interaction_params[parent][cat_val] = init_betas_by_response_mean(data_blocks[parent], 1.0)
        return interaction_masks, interaction_params

    def _init_regular_params(self, feature_groups, data_blocks, y, n_obs, init_params):
        """Initialise coefficient vectors for regular (non-interaction) groups.

        Temporarily swaps ``self.feature_groups_`` because ``infer_init_params``
        relies on it for flatten/unflatten.  A ``try/finally`` guard ensures
        the original value is always restored.
        """
        regular_groups = [g for g in feature_groups if g not in self.interactions_]
        regular_data = {g: data_blocks[g] for g in regular_groups}
        saved_fg = self.feature_groups_
        self.feature_groups_ = {g: feature_groups[g] for g in regular_groups}
        try:
            if init_params is None:
                self.global_scalar_ = np.mean(y)
                _, params_blocks = self.infer_init_params(None, regular_data, np.ones(n_obs))
            else:
                _, params_blocks = self.infer_init_params(init_params, regular_data, y)
        finally:
            self.feature_groups_ = saved_fg
        return params_blocks

    def _init_block_preds(self, data_blocks, params_blocks, interaction_params, interaction_masks):
        """Compute initial single-block predictions for every group."""
        block_preds = {}
        for key in self.feature_groups_:
            if key in self.interactions_:
                block_preds[key] = self._build_interaction_block_pred(
                    key, data_blocks[key], interaction_params, interaction_masks)
            else:
                block_preds[key] = self.forward(
                    params_blocks={key: params_blocks[key]},
                    X_blocks={key: data_blocks[key]},
                    ignore_global_scale=True)
        return block_preds

    # ------------------------------------------------------------------
    # fit — BCD step methods
    # ------------------------------------------------------------------

    def _step_interaction_group(self, group, data_blocks, block_preds,
                                interaction_params, interaction_masks, y, weights):
        """One BCD pass for an interaction group: per-category OLS, normalize, update.

        Mutates *interaction_params*, *block_preds*, and ``self.global_scalar_`` in place.
        """
        X_basis = data_blocks[group]
        fixed = vector_product_numexpr_dict_values(block_preds, exclude=group)
        mX = X_basis * fixed[:, None]

        for cat_val, cat_coef in interaction_params[group].items():
            if hasattr(cat_coef, 'predict'):
                continue
            mask = interaction_masks[group][cat_val]
            interaction_params[group][cat_val] = ols_normal_equation(
                (self.global_scalar_ * mX)[mask],
                np.asarray(y)[mask],
                weights=weights[mask] if weights is not None else None)

        combined = self._build_interaction_block_pred(group, X_basis, interaction_params, interaction_masks)
        mean = self._absorb_block_mean(combined, weights)

        if not np.isclose(mean, 0):
            for cat_val in interaction_params[group]:
                coef = interaction_params[group][cat_val]
                if not hasattr(coef, 'predict'):
                    interaction_params[group][cat_val] = coef / mean
            combined = self._build_interaction_block_pred(group, X_basis, interaction_params, interaction_masks)

        block_preds[group] = combined

    def _step_regular_group(self, group, data_blocks, block_preds, params_blocks,
                            y, weights, cache_qr_decomp, use_svd):
        """One BCD pass for a regular group: OLS, normalize, update.

        Mutates *params_blocks*, *block_preds*, and ``self.global_scalar_`` in place.
        """
        X_basis = data_blocks[group]
        fixed = vector_product_numexpr_dict_values(block_preds, exclude=group)

        if not cache_qr_decomp:
            mX = X_basis * fixed[:, None]
            if use_svd:
                if weights is not None:
                    raise NotImplementedError("Weighted least squares with SVD is not implemented yet.")
                new_params = np.linalg.lstsq(self.global_scalar_ * mX, y, rcond=None)[0]
            else:
                new_params = ols_normal_equation(self.global_scalar_ * mX, y, weights=weights)
        else:
            if weights is not None:
                raise NotImplementedError("Weighted least squares with cached QR decomposition is not implemented yet.")
            if group not in self.qr_decomp_cache_:
                Q, R = np.linalg.qr(X_basis)
                self.qr_decomp_cache_[group] = (Q, R)
            else:
                Q, R = self.qr_decomp_cache_[group]
            scaled_y = y / self.global_scalar_ / fixed
            new_params = np.linalg.solve(R, Q.T @ scaled_y)
            mX = X_basis * fixed[:, None]

        mean = self._absorb_block_mean(mX @ new_params, weights)
        if not np.isclose(mean, 0):
            new_params /= mean
        else:
            print(f"Warning: floating mean is close to zero for feature group {group}. Skipping normalization.")

        params_blocks[group] = new_params
        block_preds[group] = self.forward(
            params_blocks={group: new_params},
            X_blocks={group: data_blocks[group]},
            ignore_global_scale=True)

    # ------------------------------------------------------------------
    # fit — iteration bookkeeping
    # ------------------------------------------------------------------

    def _record_iteration(self, iteration, n_iterations, block_preds,
                          params_blocks, interaction_params, y, verbose):
        """Compute loss, snapshot state, update best, optionally print."""
        predictions = vector_product_numexpr_dict_values(block_preds) * self.global_scalar_
        loss = self.loss_function(predictions, y)

        self.loss_history_.append(loss)
        self.coef_history_.append(copy.deepcopy(params_blocks))
        self.interaction_params_history_.append(copy.deepcopy(interaction_params))
        self.global_scalar_history_.append(self.global_scalar_)

        if loss <= self.best_loss_:
            self.best_loss_ = loss
            self.best_params_ = copy.deepcopy(params_blocks)
            self.best_interaction_params_ = copy.deepcopy(interaction_params)
            self.best_iteration_ = iteration

        if verbose > 0:
            print(f"Iteration {iteration+1}/{n_iterations}, Loss: {loss:.6e}")

    def _check_convergence(self, iteration, force_rounds, early_stopping_rounds, ftol):
        """Return True if BCD should stop early."""
        if force_rounds is not None and iteration + 1 < force_rounds:
            return False

        if early_stopping_rounds is not None and len(self.loss_history_) > early_stopping_rounds:
            if self.loss_history_[-1] >= self.loss_history_[-early_stopping_rounds]:
                print(f"Early stopping at iteration {iteration+1} with Loss: {self.loss_history_[-1]:.4e}")
                return True

        if ftol is not None and len(self.loss_history_) >= 5:
            if abs(self.loss_history_[-1] / self.loss_history_[-5]) > 1 - ftol:
                print(f"Converged at iteration {iteration+1} with Loss: {self.loss_history_[-1]:.4e}")
                return True

        return False

    # ------------------------------------------------------------------
    # fit — finalization
    # ------------------------------------------------------------------

    def _finalize_interaction_scalars(self, data_blocks, interaction_masks, y, weights):
        """Post-convergence: compute per-category A/E scalars, store InteractionCoef in coef_."""
        if not self.interactions_:
            return

        final_interaction_params = copy.deepcopy(self.interaction_params_history_[-1])

        final_block_preds = {}
        for key in self.feature_groups_:
            if key not in self.interactions_:
                final_block_preds[key] = data_blocks[key] @ self.coef_[key]
            else:
                final_block_preds[key] = self._build_interaction_block_pred(
                    key, data_blocks[key], final_interaction_params, interaction_masks)

        for parent, cat_coefs in final_interaction_params.items():
            fixed = self.global_scalar_ * vector_product_numexpr_dict_values(
                final_block_preds, exclude=parent)
            scalars = {}
            for cat_val, cat_coef in cat_coefs.items():
                if hasattr(cat_coef, 'predict'):
                    scalars[cat_val] = 1.0
                    continue
                mask = interaction_masks[parent][cat_val]
                cat_pred = data_blocks[parent][mask] @ cat_coef
                y_hat_c = fixed[mask] * cat_pred
                y_c = y[mask]
                if weights is not None:
                    scalars[cat_val] = np.sum(weights[mask] * y_c) / np.sum(weights[mask] * y_hat_c)
                else:
                    scalars[cat_val] = np.sum(y_c) / np.sum(y_hat_c)

            self.coef_[parent] = InteractionCoef(
                by=self.interactions_[parent],
                categories=dict(cat_coefs),
                scalars=scalars,
            )

    def _compute_block_means(self, data_blocks, X):
        """Recompute ``block_means_`` from final ``coef_`` (including A/E scalars for interactions)."""
        for key in self.feature_groups_:
            if key in self.interactions_:
                coef = self.coef_[key]
                block_pred = self._predict_group(key, data_blocks[key], cat_series=X.orig[coef.by])
            else:
                block_pred = data_blocks[key] @ self.coef_[key]
            self.block_means_[key] = np.mean(block_pred)

    # ------------------------------------------------------------------
    # fit (public entry point)
    # ------------------------------------------------------------------

    @memorize_fit_args
    def fit( self, X: ProductModelDataContainer, feature_groups: Dict, submodels: Dict=None,
             interactions: Dict=None, interaction_submodels: Dict=None,
             init_params=None, early_stopping_rounds=5, n_iterations=20, force_rounds=5, verbose=1, ftol=1e-5,
             cache_qr_decomp=False, offset_y=None, use_svd=False, weights=None ):

        # ---- setup ----
        self._reset_history( cache_qr_decomp=cache_qr_decomp )
        self.feature_groups_ = feature_groups
        self.submodels_ = submodels or {}
        self.interactions_ = interactions or {}

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

        # data_blocks      : {str: ndarray(n_obs, n_basis)}  — expanded design matrices, read-only
        # params_blocks     : {str: ndarray(n_basis,)}        — regular-group coefficients, updated per BCD pass
        # interaction_params: {str: {cat: ndarray(n_basis,)}} — per-category coefficients for interaction groups
        # interaction_masks : {str: {cat: bool(n_obs,)}}      — row masks per category, set once
        # block_preds       : {str: ndarray(n_obs,)}          — cached block predictions (no scalar), updated per BCD pass
        # y                 : ndarray(n_obs,)                  — response (with offset if any)
        # weights           : ndarray(n_obs,) | None           — observation weights
        #
        # Full prediction = global_scalar_ * product(block_preds.values())
        interaction_masks, interaction_params = self._init_interactions(
            X, data_blocks, feature_groups, interaction_submodels or {})
        params_blocks = self._init_regular_params(
            feature_groups, data_blocks, y, X.shape[0], init_params)
        block_preds = self._init_block_preds(
            data_blocks, params_blocks, interaction_params, interaction_masks)

        # ---- BCD iterations ----
        for i in range(n_iterations):
            for group in feature_groups:
                if group in self.interactions_:
                    self._step_interaction_group(
                        group, data_blocks, block_preds, interaction_params, interaction_masks, y, weights)
                elif group not in self.submodels_:
                    self._step_regular_group(
                        group, data_blocks, block_preds, params_blocks, y, weights, cache_qr_decomp, use_svd)

            self._record_iteration(i, n_iterations, block_preds, params_blocks, interaction_params, y, verbose)
            if self._check_convergence(i, force_rounds, early_stopping_rounds, ftol):
                break

        # ---- finalize ----
        self.coef_ = copy.deepcopy(self.coef_history_[-1])
        self.global_scalar_ = self.global_scalar_history_[-1]
        self._finalize_interaction_scalars(data_blocks, interaction_masks, y, weights)
        self._compute_block_means(data_blocks, X)
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
            X_block = get_block(group)
            coef = self.coef_.get(group)
            if coef is None:
                continue
            if isinstance(coef, InteractionCoef):
                result *= self._predict_group(group, X_block, cat_series=orig[coef.by])
            elif hasattr(self, 'submodels_') and group in self.submodels_:
                result *= self.submodels_[group].predict(X_block)
            else:
                result *= X_block @ coef

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
        return self._predict_group(group_to_include, get_block(group_to_include), cat_series=orig[coef.by])

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
            X_block = get_block(group)
            coef = self.coef_.get(group)
            if coef is None:
                continue
            if isinstance(coef, InteractionCoef):
                result *= self._predict_group(group, X_block, cat_series=orig[coef.by])
            elif hasattr(self, 'submodels_') and group in self.submodels_:
                result *= self.submodels_[group].predict(X_block)
            else:
                result *= X_block @ coef

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
