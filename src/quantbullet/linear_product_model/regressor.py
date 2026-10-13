import inspect
import numpy as np
import pandas as pd
import time

from typing import Dict
from scipy.optimize import least_squares

from .base import IterativeFitMixin, LinearProductRegressorBase, InteractionCoef, memorize_fit_args
from .datacontainer import ProductModelDataContainer
from .forms import resolve_form
from .solvers import BCDSolver, NewtonSolver


def _is_default(value, default):
    """Whether a ``fit`` argument kept its default; an empty dict counts as ``None``."""
    if default is None:
        return value is None or (isinstance(value, dict) and not value)
    return not isinstance(value, np.ndarray) and value == default


class LinearProductRegressor( LinearProductRegressorBase, IterativeFitMixin ):
    """Multiplicative factor model ``y_hat = global_scalar * prod_k f_k``, one factor per feature group.

    ``fit(block_link=...)`` chooses the factor form: ``"identity"`` gives linear factors
    ``f_k = X_k @ beta_k`` (fitted block by block with BCD), ``"log"`` gives log-linear factors
    ``f_k = exp(X_k @ beta_k)`` (a log-link GLM, fitted jointly with Newton). Either way the fitted
    blocks are normalized to mean 1 and read the same way by ``predict``, the toolkit and reports.
    """

    # Solvers ``fit`` hands the iteration to, chosen by the form's ``default_solver``; a subclass may swap in variants.
    bcd_solver = BCDSolver
    newton_solver = NewtonSolver

    def __init__(self):
        LinearProductRegressorBase.__init__(self)
        IterativeFitMixin.__init__(self)
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

    def _solver_class(self, form):
        return {"bcd": self.bcd_solver, "newton": self.newton_solver}[form.default_solver]

    @memorize_fit_args
    def fit( self, X: ProductModelDataContainer, feature_groups: Dict, submodels: Dict=None,
             interactions: Dict=None, interaction_submodels: Dict=None,
             init_params=None, early_stopping_rounds=5, n_iterations=20, force_rounds=5, verbose=1, ftol=1e-5,
             cache_qr_decomp=False, offset_y=None, use_svd=False, weights=None, loss: str = 'mse',
             block_link: str = 'identity', solver_options: Dict = None ):
        """Fit the model. ``loss`` and ``block_link`` select the form, and the form selects the solver.

        ``block_link='identity'`` (default) fits blocks ``X_k @ beta_k`` by BCD, configured by
        ``init_params``, ``interaction_submodels``, ``n_iterations``, ``early_stopping_rounds``,
        ``force_rounds``, ``ftol``, ``cache_qr_decomp`` and ``use_svd``. ``block_link='log'`` fits
        blocks ``exp(X_k @ beta_k)`` jointly by Newton (Poisson loss), configured by ``solver_options``
        (see ``solvers.NewtonSolver``). A solver argument the chosen solver does not use must keep its
        default.
        """
        fit_started = time.perf_counter()
        # ---- 1) choose the form and its solver; reject what they do not support before the model is reset ----
        form = resolve_form(block_link, loss)
        solver_class = self._solver_class(form)
        arguments = dict(init_params=init_params, interaction_submodels=interaction_submodels,
                         n_iterations=n_iterations, early_stopping_rounds=early_stopping_rounds,
                         force_rounds=force_rounds, ftol=ftol, cache_qr_decomp=cache_qr_decomp, use_svd=use_svd,
                         solver_options=solver_options)
        unused = [name for name, value in arguments.items()
                  if name not in solver_class.FIT_ARGUMENTS and not _is_default(value, _FIT_DEFAULTS[name])]
        if unused:
            raise ValueError(f"{', '.join(unused)} do not apply to the {solver_class.name} solver "
                             f"of form {form.name}")
        solver = solver_class(self, form, verbose=verbose,
                              **{name: arguments[name] for name in solver_class.FIT_ARGUMENTS})
        if X.response is not None:
            form.check_response(X.response if offset_y is None else X.response + offset_y)

        # ---- 2) setup model state ----
        self.loss_, self.block_link_ = loss, block_link
        self._reset_history( cache_qr_decomp=cache_qr_decomp )
        self.feature_groups_ = feature_groups
        self.submodels_ = submodels or {}
        self.interactions_ = interactions or {}

        for parent, cat_var in self.interactions_.items():
            if cat_var not in feature_groups:
                raise ValueError(
                    f"Interaction by-variable '{cat_var}' must be included in feature_groups "
                    f"as a separate categorical block (required for '{parent}' interaction)."
                )

        # ---- 3) validate inputs and build canonical arrays ----
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

        # ---- 4) iterate ----
        solver.run(X, data_blocks, y, weights, started=fit_started)
        self.solver_name_ = self.fit_metadata_.solver = solver.name
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


# Defaults of ``fit``'s solver arguments, read from its signature so the two cannot drift apart.
_FIT_DEFAULTS = {name: parameter.default
                 for name, parameter in inspect.signature(LinearProductRegressor.fit).parameters.items()}

# Former name, kept permanently: saved models are unpickled by this module path and class name.
LinearProductRegressorBCD = LinearProductRegressor


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
