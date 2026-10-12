"""Solvers for the linear product model.

A solver owns the iteration of a fit and none of the formulas: block updates come from the model's
form (``forms.ModelForm``), and the fitted state is written to the model the solver was built for --
``coef_``, ``global_scalar_``, the histories, ``best_*``, ``convergence_info_`` and the timings in
``fit_timing_`` (created by ``fit``). A solver keeps no per-row data after ``run``.
"""
import time

import numpy as np

from .base import InteractionCoef
from .utils import init_betas_by_response_mean
from ._acceleration import one_hot_codes, vector_product_numexpr_dict_values


class BCDSolver:
    """Block coordinate descent for identity forms.

    Each sweep updates one block at a time with the form's ``block_step`` while the other blocks stay
    fixed, then rescales the block to mean 1 and carries the mean in ``global_scalar_``; with an
    identity link that divides the block's coefficients by its mean. Interaction curves are fitted per
    category; their levels move to the ``by`` block when it is a trainable full one-hot block. The
    sweep with the lowest loss is restored at the end.
    """

    name = "bcd"

    def __init__(self, model, *, n_iterations=20, early_stopping_rounds=5, force_rounds=5, ftol=1e-5,
                 cache_qr_decomp=False, use_svd=False, verbose=1):
        form = model.form_
        if form.link != "identity" or not hasattr(form, "block_step"):
            raise ValueError(f"BCD needs an identity form with block_step; got {form.name}")
        self.model, self.form = model, form
        self.n_iterations, self.early_stopping_rounds = n_iterations, early_stopping_rounds
        self.force_rounds, self.ftol, self.verbose = force_rounds, ftol, verbose
        self.cache_qr_decomp, self.use_svd = cache_qr_decomp, use_svd

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, X, data_blocks, y, weights, init_params, interaction_submodels, started):
        """Fit ``self.model`` on container ``X``; ``started`` is when ``fit`` began, for ``setup_seconds``."""
        model = self.model
        feature_groups = model.feature_groups_
        self._check_step_options(weights)

        # ---- initialize global level and block-level states ----
        # data_blocks       : {str: ndarray(n_obs, n_basis)}   expanded design matrices (read-only)
        # interaction_masks : {str: {cat: bool(n_obs,)}}       per-category row masks
        # interaction_params: {str: {cat: ndarray(n_basis,)}}  interaction-group coefficients
        # params_blocks     : {str: ndarray(n_basis,)}         regular-group coefficients
        # block_preds       : {str: ndarray(n_obs,)}           cached block predictions (without scalar)
        # y                 : ndarray(n_obs,)                  response (offset-adjusted if needed)
        # weights           : ndarray(n_obs,) | None           optional observation weights
        # full_prediction   : global_scalar_ * product(block_preds.values())
        self._init_global_scalar(y, init_params, weights)
        interaction_masks, interaction_params = self._init_interactions(
            X.orig, data_blocks, feature_groups, interaction_submodels)
        params_blocks = self._infer_regular_params(
            feature_groups, data_blocks, y, init_params)
        block_preds = self._init_block_preds(
            data_blocks, params_blocks, interaction_params, interaction_masks)
        interaction_level_columns = {
            group: self._interaction_level_columns(
                group, data_blocks, interaction_masks, params_blocks)
            for group in model.interactions_
        }

        y_arr = np.asarray(y)
        interaction_data_cache = {}
        for group in model.interactions_:
            interaction_data_cache[group] = {}
            for cat_val, mask in interaction_masks[group].items():
                interaction_data_cache[group][cat_val] = {
                    'X': data_blocks[group][mask],
                    'y': y_arr[mask],
                    'w': weights[mask] if weights is not None else None,
                }

        categorical_codes = self._init_one_hot_codes(data_blocks, params_blocks)
        model.fit_timing_['setup_seconds'] = time.perf_counter() - started

        # ---- BCD iterations ----
        model.global_scalar_step_history_.append((0, '_init', model.global_scalar_))
        for i in range(self.n_iterations):
            sweep_started = time.perf_counter()
            for group in feature_groups:
                block_started = time.perf_counter()
                if group in model.interactions_:
                    self._step_interaction_group(
                        group, data_blocks, block_preds, interaction_params, interaction_masks, y, weights,
                        interaction_data_cache=interaction_data_cache, params_blocks=params_blocks,
                        interaction_level_columns=interaction_level_columns)
                elif group not in model.submodels_:
                    self._step_regular_group(
                        group, data_blocks, block_preds, params_blocks, y, weights,
                        one_hot_codes=categorical_codes.get(group))
                model.fit_timing_['block_seconds'][group] += time.perf_counter() - block_started
                model.global_scalar_step_history_.append((i + 1, group, model.global_scalar_))

            self._record_iteration(i, block_preds, params_blocks, interaction_params, y, weights)
            conv_info = self._check_convergence(i)
            model.fit_timing_['sweep_seconds'].append(time.perf_counter() - sweep_started)
            if conv_info is not None:
                model.convergence_info_ = conv_info
                if self.verbose > 0:
                    print(self._format_convergence_info(conv_info))
                break
        else:
            model.convergence_info_ = {
                'reason': 'max_iterations',
                'iteration': self.n_iterations,
                'current_loss': model.loss_history_[-1],
                'best_iteration': model.best_iteration_ + 1,
                'best_loss': model.best_loss_,
            }
            if self.verbose > 0:
                print(self._format_convergence_info(model.convergence_info_))

        # ---- finalize: restore best iteration ----
        model.coef_ = self._snapshot_params(model.best_params_)
        model.global_scalar_ = model.global_scalar_history_[model.best_iteration_]
        self._store_interaction_coefs()
        model._compute_block_means(data_blocks, X, weights)
        model.fit_timing_['total_seconds'] = time.perf_counter() - started

    # ------------------------------------------------------------------
    # initialisation
    # ------------------------------------------------------------------

    def _check_step_options(self, weights):
        """Reject weighted QR / SVD solves before any work, as ``block_step`` would on the first block."""
        if weights is None:
            return
        if self.cache_qr_decomp and "qr" in self.form.step_options:
            raise NotImplementedError("Weighted least squares with cached QR decomposition is not implemented yet.")
        if self.use_svd and "use_svd" in self.form.step_options:
            raise NotImplementedError("Weighted least squares with SVD is not implemented yet.")

    def _init_interactions(self, orig_df, data_blocks, feature_groups, interaction_submodels):
        """Build ``interaction_masks`` and ``interaction_params`` from raw categorical columns."""
        interaction_masks = {}
        interaction_params = {}
        for parent, cat_var in self.model.interactions_.items():
            if parent not in feature_groups:
                raise ValueError(f"Interaction parent '{parent}' not found in feature_groups.")
            categories = sorted(orig_df[cat_var].dropna().unique(), key=str)
            parent_subs = interaction_submodels.get(parent, {})
            interaction_masks[parent] = {}
            interaction_params[parent] = {}
            for cat_val in categories:
                interaction_masks[parent][cat_val] = (orig_df[cat_var] == cat_val).values
                if cat_val in parent_subs:
                    interaction_params[parent][cat_val] = parent_subs[cat_val]
                else:
                    interaction_params[parent][cat_val] = init_betas_by_response_mean(data_blocks[parent], 1.0)
        return interaction_masks, interaction_params

    def _init_global_scalar(self, y, init_params, weights=None):
        """Initialize the model-level scalar before block-level parameter inference.

        For default initialization (``init_params is None``), block params are inferred
        around a unit target, so the response level is carried by ``global_scalar_``.
        """
        if init_params is None:
            self.model.global_scalar_ = np.average(y, weights=weights) if weights is not None else np.mean(y)

    def _infer_regular_params(self, feature_groups, data_blocks, y, init_params):
        """Infer coefficient vectors for regular (non-interaction) groups."""
        regular_data = {
            group: data_blocks[group]
            for group in feature_groups
            if group not in self.model.interactions_
        }
        target_y = np.ones_like(y, dtype=float) if init_params is None else y
        _, params_blocks = self.model.infer_init_params(init_params, regular_data, target_y)
        return params_blocks

    def _init_block_preds(self, data_blocks, params_blocks, interaction_params, interaction_masks):
        """Compute initial single-block predictions for every group."""
        model = self.model
        block_preds = {}
        for key in model.feature_groups_:
            if key in model.interactions_:
                block_preds[key] = self._build_interaction_block_pred(
                    key, data_blocks[key], interaction_params, interaction_masks)
            else:
                block_preds[key] = model.forward(
                    params_blocks={key: params_blocks[key]},
                    X_blocks={key: data_blocks[key]},
                    ignore_global_scale=True)
        return block_preds

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
            the factor of its category's coefficients.
        """
        combined = np.ones(data_block.shape[0], dtype=float)
        for cat_val, cat_coef in interaction_params[group].items():
            m = masks[group][cat_val]
            combined[m] = self.model._coef_factor(cat_coef, data_block[m])
        return combined

    def _interaction_level_columns(self, group, data_blocks, interaction_masks, params_blocks):
        """Map category labels to actual full-one-hot columns, when levels can move."""
        by = self.model.interactions_[group]
        if by not in params_blocks or by in getattr(self.model, 'submodels_', {}):
            return None
        columns = {}
        for cat_val, mask in interaction_masks[group].items():
            X_c = data_blocks[by][mask]
            active = np.flatnonzero(np.any(X_c != 0, axis=0))
            if len(active) != 1 or not np.all(X_c[:, active[0]] == 1):
                return None
            columns[cat_val] = int(active[0])
        if len(set(columns.values())) != len(columns):
            return None
        return columns

    def _init_one_hot_codes(self, data_blocks, params_blocks):
        """Fit-local codes for forms with a grouped one-hot step; other blocks keep the dense step."""
        if not self.form.grouped_one_hot:
            return {}
        codes = {}
        for group in params_blocks:
            if group not in self.model.submodels_:
                group_codes = one_hot_codes(data_blocks[group])
                if group_codes is not None:
                    codes[group] = group_codes
        return codes

    # ------------------------------------------------------------------
    # block steps
    # ------------------------------------------------------------------

    def _absorb_block_mean(self, block_pred, weights=None):
        """Compute (weighted) mean of *block_pred*; if non-zero, absorb into ``global_scalar_``.

        Returns the mean so the caller can decide whether to divide params.
        """
        mean = np.average(block_pred, weights=weights) if weights is not None else np.mean(block_pred)
        if not np.isclose(mean, 0):
            self.model.global_scalar_ *= mean
        return mean

    def _step_options(self, group, X_basis):
        """The optional solve variants requested for ``fit`` that this form supports."""
        options = {}
        if self.cache_qr_decomp and "qr" in self.form.step_options:
            cache = self.model.qr_decomp_cache_
            if group not in cache:
                cache[group] = np.linalg.qr(X_basis)
            options['qr'] = cache[group]
        if self.use_svd and "use_svd" in self.form.step_options:
            options['use_svd'] = True
        return options

    def _step_interaction_group(self, group, data_blocks, block_preds,
                                interaction_params, interaction_masks, y, weights,
                                interaction_data_cache=None, params_blocks=None,
                                interaction_level_columns=None):
        """One BCD pass for an interaction group: per-category block step, normalize, update.

        Transfer each fitted curve's mean to its categorical coefficient before
        normalizing the categorical block into ``global_scalar_``. Both operations
        preserve the full candidate prediction. If the categorical block cannot
        absorb independent levels, keep the fitted curves at their original scale.
        """
        model = self.model
        X_basis = data_blocks[group]
        fixed = vector_product_numexpr_dict_values(block_preds, exclude=group)
        combined = np.ones(X_basis.shape[0], dtype=float)
        level_columns = (interaction_level_columns or {}).get(group)
        level_scales = {}

        for cat_val, cat_coef in interaction_params[group].items():
            mask = interaction_masks[group][cat_val]

            if hasattr(cat_coef, 'predict'):
                X_c = interaction_data_cache[group][cat_val]['X'] if interaction_data_cache else X_basis[mask]
                combined[mask] = cat_coef.predict(X_c)
                continue

            if interaction_data_cache is not None:
                cached = interaction_data_cache[group][cat_val]
                X_c, y_c, w_c = cached['X'], cached['y'], cached['w']
            else:
                X_c = X_basis[mask]
                y_c = np.asarray(y)[mask]
                w_c = weights[mask] if weights is not None else None

            new_coef = self.form.block_step(X_c, y_c, model.global_scalar_, fixed[mask], w_c, cat_coef)

            cat_pred = X_c @ new_coef
            cat_mean = np.average(cat_pred, weights=w_c) if w_c is not None else np.mean(cat_pred)
            if level_columns is not None and not np.isclose(cat_mean, 0):
                new_coef = new_coef / cat_mean
                combined[mask] = cat_pred / cat_mean
                level_scales[level_columns[cat_val]] = cat_mean
            else:
                combined[mask] = cat_pred
            interaction_params[group][cat_val] = new_coef

        block_preds[group] = combined

        # Apply all transfers together: every category above used the same fixed
        # components and global scalar for its candidate update.
        if level_scales:
            by = model.interactions_[group]
            by_coef = params_blocks[by].copy()
            for column, scale in level_scales.items():
                by_coef[column] *= scale
            by_pred = data_blocks[by] @ by_coef
            mean = self._absorb_block_mean(by_pred, weights)
            if not np.isclose(mean, 0):
                by_coef /= mean
                by_pred = by_pred / mean
            params_blocks[by] = by_coef
            block_preds[by] = by_pred

    def _step_regular_group(self, group, data_blocks, block_preds, params_blocks,
                            y, weights, one_hot_codes=None):
        """One BCD pass for a regular group: block step, normalize, update.

        Mutates *params_blocks*, *block_preds*, and the model's ``global_scalar_`` in place.
        """
        X_basis = data_blocks[group]
        fixed = vector_product_numexpr_dict_values(block_preds, exclude=group)
        new_params = self.form.block_step(X_basis, y, self.model.global_scalar_, fixed, weights,
                                          params_blocks[group], codes=one_hot_codes,
                                          **self._step_options(group, X_basis))

        raw_pred = new_params[one_hot_codes] if one_hot_codes is not None else X_basis @ new_params
        mean = self._absorb_block_mean(raw_pred, weights)
        if not np.isclose(mean, 0):
            new_params /= mean
            block_preds[group] = raw_pred / mean
        else:
            print(f"Warning: floating mean is close to zero for feature group {group}. Skipping normalization.")
            block_preds[group] = raw_pred

        params_blocks[group] = new_params

    # ------------------------------------------------------------------
    # iteration bookkeeping
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_params(params_blocks):
        return {k: v.copy() for k, v in params_blocks.items()}

    @staticmethod
    def _snapshot_interaction_params(interaction_params):
        return {
            parent: {cat: (c.copy() if isinstance(c, np.ndarray) else c)
                     for cat, c in cats.items()}
            for parent, cats in interaction_params.items()
        }

    def _record_iteration(self, iteration, block_preds, params_blocks, interaction_params, y, weights):
        """Compute loss, snapshot state, update best, optionally print."""
        model = self.model
        predictions = vector_product_numexpr_dict_values(block_preds) * model.global_scalar_
        loss = model.loss_function(predictions, y, weights=weights)

        model.loss_history_.append(loss)
        model.coef_history_.append(self._snapshot_params(params_blocks))
        model.interaction_params_history_.append(self._snapshot_interaction_params(interaction_params))
        model.global_scalar_history_.append(model.global_scalar_)

        if loss <= model.best_loss_:
            model.best_loss_ = loss
            model.best_params_ = self._snapshot_params(params_blocks)
            model.best_interaction_params_ = self._snapshot_interaction_params(interaction_params)
            model.best_iteration_ = iteration

        if self.verbose > 0:
            print(f"Iteration {iteration+1}/{self.n_iterations}, Loss: {loss:.6e}")

    def _check_convergence(self, iteration):
        """Check whether BCD should stop early.

        Returns a dict describing the termination reason, or ``None`` to
        continue.  The dict is stored as the model's ``convergence_info_``.
        """
        model = self.model
        if self.force_rounds is not None and iteration + 1 < self.force_rounds:
            return None

        cur_loss = model.loss_history_[-1]

        if self.early_stopping_rounds is not None and len(model.loss_history_) > self.early_stopping_rounds:
            ref_loss = model.loss_history_[-self.early_stopping_rounds]
            if cur_loss >= ref_loss:
                return {
                    'reason': 'early_stopping',
                    'iteration': iteration + 1,
                    'current_loss': cur_loss,
                    'reference_loss': ref_loss,
                    'lookback': self.early_stopping_rounds,
                    'best_iteration': model.best_iteration_ + 1,
                    'best_loss': model.best_loss_,
                }

        if self.ftol is not None and len(model.loss_history_) >= 5:
            ref_loss = model.loss_history_[-5]
            rel_improvement = 1 - cur_loss / ref_loss
            if 0 <= rel_improvement < self.ftol:
                return {
                    'reason': 'ftol',
                    'iteration': iteration + 1,
                    'current_loss': cur_loss,
                    'reference_loss': ref_loss,
                    'rel_improvement': rel_improvement,
                    'ftol': self.ftol,
                }

        return None

    @staticmethod
    def _format_convergence_info(info):
        """Format a convergence_info_ dict into a human-readable string."""
        if info is None:
            return "Model has not converged yet."
        reason = info['reason']
        if reason == 'early_stopping':
            return (
                f"Early stopping at iteration {info['iteration']}: "
                f"current loss {info['current_loss']:.6e} >= loss {info['reference_loss']:.6e} "
                f"from {info['lookback']} iterations ago (no improvement). "
                f"Best iteration was {info['best_iteration']} with loss {info['best_loss']:.6e}."
            )
        if reason == 'ftol':
            return (
                f"Converged at iteration {info['iteration']}: "
                f"relative improvement over last 5 iterations = {info['rel_improvement']:.2e} "
                f"< ftol={info['ftol']:.1e} "
                f"(loss {info['reference_loss']:.6e} -> {info['current_loss']:.6e})."
            )
        if reason == 'max_iterations':
            return (
                f"Reached maximum iterations ({info['iteration']}). "
                f"Final loss: {info['current_loss']:.6e}. "
                f"Best iteration was {info['best_iteration']} with loss {info['best_loss']:.6e}."
            )
        return str(info)

    # ------------------------------------------------------------------
    # finalization
    # ------------------------------------------------------------------

    def _store_interaction_coefs(self):
        """Store best interaction params as InteractionCoef in coef_."""
        model = self.model
        if not model.interactions_:
            return
        snapshot = self._snapshot_interaction_params(model.best_interaction_params_)
        for parent, cat_coefs in snapshot.items():
            model.coef_[parent] = InteractionCoef(
                by=model.interactions_[parent],
                categories=dict(cat_coefs),
            )
