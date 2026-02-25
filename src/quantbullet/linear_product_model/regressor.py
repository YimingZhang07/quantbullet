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
    # Interaction helpers
    # ------------------------------------------------------------------

    def _build_interaction_block_pred(self, group, data_block, interaction_params, masks):
        """Assemble the combined prediction vector for an interaction group."""
        combined = np.ones(data_block.shape[0], dtype=float)
        for cat_val, cat_coef in interaction_params[group].items():
            m = masks[group][cat_val]
            if hasattr(cat_coef, 'predict'):
                combined[m] = cat_coef.predict(data_block[m])
            else:
                combined[m] = data_block[m] @ cat_coef
        return combined

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    @memorize_fit_args
    def fit( self, X: ProductModelDataContainer, feature_groups: Dict, submodels: Dict=None,
             interactions: Dict=None, interaction_submodels: Dict=None,
             init_params=None, early_stopping_rounds=5, n_iterations=20, force_rounds=5, verbose=1, ftol=1e-5,
             cache_qr_decomp=False, offset_y=None, use_svd=False, weights=None ):

        self._reset_history( cache_qr_decomp=cache_qr_decomp )
        self.feature_groups_ = feature_groups
        self.submodels_ = submodels or {}
        self.interactions_ = interactions or {}
        interaction_submodels = interaction_submodels or {}

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

        n_obs = X.shape[0]

        # ---------- interaction metadata: categories & boolean masks ----------
        interaction_masks = {}   # {parent: {cat_val: bool_array}}
        interaction_params = {}  # {parent: {cat_val: ndarray | submodel}}
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

        # ---------- initialise regular-group params ----------
        regular_groups = [g for g in feature_groups if g not in self.interactions_]
        regular_data = {g: data_blocks[g] for g in regular_groups}

        if init_params is None:
            self.global_scalar_ = np.mean(y)
            saved_fg = self.feature_groups_
            self.feature_groups_ = {g: feature_groups[g] for g in regular_groups}
            _, params_blocks = self.infer_init_params(None, regular_data, np.ones(n_obs))
            self.feature_groups_ = saved_fg
        else:
            saved_fg = self.feature_groups_
            self.feature_groups_ = {g: feature_groups[g] for g in regular_groups}
            _, params_blocks = self.infer_init_params(init_params, regular_data, y)
            self.feature_groups_ = saved_fg

        # ---------- initial block predictions ----------
        block_preds = {}
        for key in feature_groups:
            if key in self.interactions_:
                block_preds[key] = self._build_interaction_block_pred(
                    key, data_blocks[key], interaction_params, interaction_masks)
            else:
                block_preds[key] = self.forward(
                    params_blocks={key: params_blocks[key]},
                    X_blocks={key: data_blocks[key]},
                    ignore_global_scale=True)

        # ========================== BCD iterations ==========================
        for i in range(n_iterations):
            for feature_group in feature_groups:

                # ---------- interaction group ----------
                if feature_group in self.interactions_:
                    floating_data = data_blocks[feature_group]
                    fixed_predictions = vector_product_numexpr_dict_values(block_preds, exclude=feature_group)
                    mX = floating_data * fixed_predictions[:, None]

                    for cat_val, cat_coef in interaction_params[feature_group].items():
                        if hasattr(cat_coef, 'predict'):
                            continue  # frozen submodel
                        bool_mask = interaction_masks[feature_group][cat_val]
                        eff_w = weights * bool_mask if weights is not None else bool_mask.astype(np.float64)
                        interaction_params[feature_group][cat_val] = ols_normal_equation(
                            self.global_scalar_ * mX, y, weights=eff_w)

                    combined = self._build_interaction_block_pred(
                        feature_group, floating_data, interaction_params, interaction_masks)
                    combined_mean = np.mean(combined)

                    if not np.isclose(combined_mean, 0):
                        for cat_val, cat_coef in interaction_params[feature_group].items():
                            if not hasattr(cat_coef, 'predict'):
                                interaction_params[feature_group][cat_val] = cat_coef / combined_mean
                        self.global_scalar_ *= combined_mean
                        combined = self._build_interaction_block_pred(
                            feature_group, floating_data, interaction_params, interaction_masks)

                    block_preds[feature_group] = combined

                # ---------- regular group ----------
                elif feature_group not in self.submodels_:
                    floating_data = data_blocks[feature_group]
                    fixed_predictions = vector_product_numexpr_dict_values(block_preds, exclude=feature_group)

                    if not cache_qr_decomp:
                        mX = floating_data * fixed_predictions[:, None]
                        if use_svd:
                            if weights is not None:
                                raise NotImplementedError("Weighted least squares with SVD is not implemented yet.")
                            floating_params = np.linalg.lstsq(self.global_scalar_ * mX, y, rcond=None)[0]
                        else:
                            floating_params = ols_normal_equation(self.global_scalar_ * mX, y, weights=weights)
                    else:
                        if weights is not None:
                            raise NotImplementedError("Weighted least squares with cached QR decomposition is not implemented yet.")
                        if feature_group not in self.qr_decomp_cache_:
                            Q, R = np.linalg.qr(floating_data)
                            self.qr_decomp_cache_[feature_group] = (Q, R)
                        else:
                            Q, R = self.qr_decomp_cache_[feature_group]
                        scaled_y = y / self.global_scalar_ / fixed_predictions
                        floating_params = np.linalg.solve(R, Q.T @ scaled_y)
                        mX = floating_data * fixed_predictions[:, None]

                    floating_predictions = mX @ floating_params
                    floating_mean = np.mean(floating_predictions)

                    if not np.isclose(floating_mean, 0):
                        floating_params /= floating_mean
                        self.global_scalar_ = self.global_scalar_ * floating_mean
                    else:
                        print(f"Warning: floating mean is close to zero for feature group {feature_group} at iteration {i}. Skipping normalization.")

                    params_blocks[feature_group] = floating_params
                    block_preds[feature_group] = self.forward(
                        params_blocks={feature_group: floating_params},
                        X_blocks={feature_group: data_blocks[feature_group]},
                        ignore_global_scale=True)

                # ---------- whole-group submodel (frozen) ----------
                else:
                    pass

            # ---------- tracking ----------
            predictions = vector_product_numexpr_dict_values(block_preds) * self.global_scalar_
            loss = self.loss_function(predictions, y)
            self.loss_history_.append(loss)
            self.coef_history_.append(copy.deepcopy(params_blocks))
            self.global_scalar_history_.append(self.global_scalar_)

            if loss <= self.best_loss_:
                self.best_loss_ = loss
                self.best_params_ = copy.deepcopy(params_blocks)
                self.best_iteration_ = i

            if verbose > 0:
                print(f"Iteration {i+1}/{n_iterations}, Loss: {loss:.6e}")

            if force_rounds is not None and i + 1 < force_rounds:
                continue

            if early_stopping_rounds is not None and len(self.loss_history_) > early_stopping_rounds:
                if self.loss_history_[-1] >= self.loss_history_[-early_stopping_rounds]:
                    print(f"Early stopping at iteration {i+1} with Loss: {loss:.4e}")
                    break

            if ftol is not None and len(self.loss_history_) >= 5:
                if abs(self.loss_history_[-1] / self.loss_history_[-5]) > 1 - ftol:
                    print(f"Converged at iteration {i+1} with Loss: {loss:.4e}")
                    break

        # ========================== finalise ==========================
        self.coef_ = copy.deepcopy(self.coef_history_[-1])
        self.global_scalar_ = self.global_scalar_history_[-1]

        for parent, cat_coefs in interaction_params.items():
            self.coef_[parent] = InteractionCoef(
                by=self.interactions_[parent],
                categories=dict(cat_coefs),
            )

        for key in feature_groups:
            if key in self.interactions_:
                self.block_means_[key] = np.mean(block_preds[key])
            else:
                block_pred = data_blocks[key] @ self.coef_[key]
                self.block_means_[key] = np.mean(block_pred)

        return self

    # ------------------------------------------------------------------
    # predict  (override to handle InteractionCoef in coef_)
    # ------------------------------------------------------------------

    def predict(self, X: ProductModelDataContainer | pd.DataFrame):
        if self.feature_groups_ is None or self.coef_ is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")

        if not self.interactions_:
            return super().predict(X)

        if isinstance(X, ProductModelDataContainer):
            orig = X.orig
            get_block = X.get_expanded_array_for_feature_group
        elif isinstance(X, pd.DataFrame):
            orig = X
            get_block = lambda g: X[self.feature_groups_[g]].values
        else:
            raise ValueError("Invalid input type. Expected ProductModelDataContainer or pd.DataFrame.")

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

        if isinstance(X, ProductModelDataContainer):
            X_block = X.get_expanded_array_for_feature_group(group_to_include)
            cat_series = X.orig[coef.by]
        elif isinstance(X, pd.DataFrame):
            X_block = X[self.feature_groups_[group_to_include]].values
            cat_series = X[coef.by]
        else:
            raise ValueError("Invalid input type.")

        return self._predict_group(group_to_include, X_block, cat_series=cat_series)

    def leave_out_feature_group_predict(self, group_to_exclude, X,
                                        params_dict=None, ignore_global_scale=False):
        if not self.interactions_:
            return super().leave_out_feature_group_predict(
                group_to_exclude, X, params_dict, ignore_global_scale)

        if group_to_exclude not in self.feature_groups_:
            raise ValueError(f"Feature group '{group_to_exclude}' not found in feature_groups_.")

        if isinstance(X, ProductModelDataContainer):
            orig = X.orig
            get_block = X.get_expanded_array_for_feature_group
        elif isinstance(X, pd.DataFrame):
            orig = X
            get_block = lambda g: X[self.feature_groups_[g]].values
        else:
            raise ValueError("Invalid input type.")

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
