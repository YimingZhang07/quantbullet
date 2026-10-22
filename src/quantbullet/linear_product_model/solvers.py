"""Solvers for the linear product model.

A solver owns the iteration of a fit and none of the formulas: updates come from the model's form
(``forms.ModelForm``), and the fitted state is written to the model the solver was built for --
``coef_``, ``global_scalar_``, the histories, ``best_*``, ``convergence_info_`` and ``fit_timing_``.
``FIT_ARGUMENTS`` lists the ``fit`` arguments a solver takes; ``fit`` passes those and rejects any
other solver argument given a non-default value. A solver keeps no per-row data after ``run``.
"""
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

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
    FIT_ARGUMENTS = ("init_params", "interaction_submodels", "n_iterations", "early_stopping_rounds",
                     "force_rounds", "ftol", "cache_qr_decomp", "use_svd")

    def __init__(self, model, form=None, *, init_params=None, interaction_submodels=None, n_iterations=20,
                 early_stopping_rounds=5, force_rounds=5, ftol=1e-5, cache_qr_decomp=False, use_svd=False,
                 verbose=1):
        form = form if form is not None else model.form_
        if form.link != "identity" or not hasattr(form, "block_step"):
            raise ValueError(f"BCD needs an identity form with block_step; got {form.name}")
        self.model, self.form = model, form
        self.init_params, self.interaction_submodels = init_params, interaction_submodels or {}
        self.n_iterations, self.early_stopping_rounds = n_iterations, early_stopping_rounds
        self.force_rounds, self.ftol, self.verbose = force_rounds, ftol, verbose
        self.cache_qr_decomp, self.use_svd = cache_qr_decomp, use_svd

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, X, data_blocks, y, weights, started):
        """Fit ``self.model`` on container ``X``; ``started`` is when ``fit`` began, for ``setup_seconds``."""
        model = self.model
        feature_groups = model.feature_groups_
        init_params, interaction_submodels = self.init_params, self.interaction_submodels
        model.fit_timing_ = {'setup_seconds': 0., 'block_seconds': {group: 0. for group in feature_groups},
                             'sweep_seconds': [], 'total_seconds': 0.}
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


# ----------------------------------------------------------------------
# Newton for log forms
# ----------------------------------------------------------------------

@dataclass
class _Part:
    """One trainable block in the Newton design."""
    group: str
    kind: str                       # "numeric", "categorical" or "interaction"
    n_block_columns: int
    columns: np.ndarray             # block columns expanded into the design
    constant: np.ndarray | None = None   # coefficients c with X @ c = 1, when the block has them
    by: str | None = None           # interaction: full one-hot block whose levels split the curve
    reference: int | None = None    # categorical: level left out of the design
    pivot: int | None = None        # block column held fixed: the left-out column or level
    pivot_from_reference: bool = False   # the pivot is the group's reference rather than chosen here
    keep: np.ndarray = None         # expanded columns kept (varying, not the reference)
    mean: np.ndarray = None
    std: np.ndarray = None
    beta: slice = None              # position in the coefficient vector; 0 is the intercept


def _constant_direction(X, w, chunk_rows, pivot=None):
    """Coefficients ``c`` with ``X @ c = 1`` and the column the design leaves out for them, or ``(None, None)``.

    A column that is identically 1 (a bias) is its own direction and is left out. Otherwise, when every row
    sums to 1 (a tent basis), ``c`` is all ones and the largest-weight column is left out, as for a categorical
    block, unless ``pivot`` (a reference column with data) is given. Row sums allow float32 rounding. ``w``
    may be ``None`` for equal weights.
    """
    ones = np.ones(X.shape[1], dtype=bool)
    rows_sum_to_one, weight = True, np.zeros(X.shape[1])
    for start in range(0, X.shape[0], chunk_rows):
        block = X[start:start + chunk_rows]
        ones &= np.all(block == 1, axis=0)
        if rows_sum_to_one:
            rows_sum_to_one = bool(np.all(np.abs(block.sum(axis=1, dtype=np.float64) - 1.) <= 1e-6))
        if rows_sum_to_one:
            weight += block.sum(axis=0, dtype=np.float64) if w is None else w[start:start + chunk_rows] @ block
        if not ones.any() and not rows_sum_to_one:
            return None, None
    if ones.any():
        column = int(np.flatnonzero(ones)[0])
        return np.eye(X.shape[1])[column], column
    if pivot is not None and weight[pivot] > 0:
        return np.ones(X.shape[1]), pivot
    return np.ones(X.shape[1]), int(np.argmax(weight))


def _without(n_columns, column):
    """``0..n_columns-1`` without ``column`` (all of them when ``column`` is None)."""
    columns = np.arange(n_columns)
    return columns if column is None else np.delete(columns, column)


_THREAD_ROWS = 1 << 16                          # rows per thread task: design assembly, history means
_THREADS = min(8, os.cpu_count() or 1)


class _LogDesign:
    """Identifiable, standardized design of a log form, assembled chunk by chunk from the model's blocks.

    There is one global intercept: numeric blocks with a constant direction lose one column along it (a
    bias column, or the largest-weight column when rows sum to 1), categorical blocks their largest-weight
    level, and an interaction block becomes its remaining columns times the full one-hot ``by`` block. A
    group whose reference falls on one column (``model.pivots_``) loses that column instead. Frozen
    submodels stay out of the design and enter as an offset.
    """

    def __init__(self, model, data_blocks, orig, w, chunk_rows):
        self.chunk_rows, self.cached = chunk_rows, None
        self.parts, self.frozen, self.codes, self.levels = [], [], {}, {}
        pivots = getattr(model, "pivots_", None) or {}
        for group in model.feature_groups_:
            X = data_blocks[group]
            if group in model.submodels_:
                self.frozen.append(group)
                continue
            if group in model.interactions_:
                by = model.interactions_[group]
                self._check_interaction(group, by, model, data_blocks, orig)
                constant, left_out = _constant_direction(X, w, chunk_rows, pivots.get(group))
                self.parts.append(_Part(group, "interaction", X.shape[1], _without(X.shape[1], left_out),
                                        constant, by=by, pivot=left_out,
                                        pivot_from_reference=left_out is not None and left_out == pivots.get(group)))
                continue
            codes = self.codes[group] if group in self.codes else one_hot_codes(X)
            if codes is not None:
                self.codes[group] = codes
                weight = np.bincount(codes, weights=w, minlength=X.shape[1])
                pivot = pivots.get(group)
                reference = pivot if pivot is not None and weight[pivot] > 0 else int(np.argmax(weight))
                self.parts.append(_Part(group, "categorical", X.shape[1], np.arange(X.shape[1]),
                                        reference=reference, pivot=reference,
                                        pivot_from_reference=reference == pivot))
                continue
            constant, left_out = _constant_direction(X, w, chunk_rows, pivots.get(group))
            self.parts.append(_Part(group, "numeric", X.shape[1], _without(X.shape[1], left_out), constant,
                                    pivot=left_out,
                                    pivot_from_reference=left_out is not None and left_out == pivots.get(group)))
        self._standardize(data_blocks, len(w))

    def _check_interaction(self, group, by, model, data_blocks, orig):
        """The ``by`` block must be a trainable full one-hot block whose columns match the labels in ``orig``."""
        codes = one_hot_codes(data_blocks[by]) if by not in model.submodels_ else None
        if codes is None:
            raise NotImplementedError(
                f"Newton fits interaction '{group}' only when its by-block '{by}' is a trainable full one-hot block")
        labels = orig[by]
        if labels.isna().any():
            raise ValueError(f"Interaction by-variable '{by}' has missing values")
        levels = {}
        for label in sorted(labels.unique(), key=str):
            columns = np.unique(codes[(labels == label).values])
            if len(columns) != 1:
                raise ValueError(f"Rows with {by}={label!r} do not map to one column of block '{by}'")
            levels[label] = int(columns[0])
        self.codes[by], self.levels[group] = codes, levels

    def chunks(self, n):
        for start in range(0, n, self.chunk_rows):
            yield slice(start, min(start + self.chunk_rows, n))

    def _expand(self, part, blocks, rows):
        X = blocks[part.group][rows]
        if part.kind == "categorical":
            return X
        if part.kind == "numeric":
            return X[:, part.columns]
        by = blocks[part.by][rows]
        return (X[:, part.columns, None] * by[:, None, :]).reshape(len(X), -1)   # column-major in levels

    def _standardize(self, blocks, n):
        sums, squares = {}, {}
        for rows in self.chunks(n):
            for part in self.parts:
                raw = self._expand(part, blocks, rows).astype(np.float64)
                sums[part.group] = sums.get(part.group, 0.) + raw.sum(axis=0)
                squares[part.group] = squares.get(part.group, 0.) + np.einsum("ij,ij->j", raw, raw)
        start = 1
        for part in self.parts:
            mean = sums[part.group] / n
            std = np.sqrt(np.maximum(squares[part.group] / n - mean ** 2, 0.))
            keep = std > 1e-12 * np.maximum(1., np.abs(mean))     # absent levels and constant pieces drop out
            if part.reference is not None:
                keep[part.reference] = False
            part.keep = np.flatnonzero(keep)
            part.mean, part.std = mean[part.keep], std[part.keep]
            part.beta = slice(start, start + len(part.keep))
            start += len(part.keep)
        self.n_columns = start

    def part_matrix(self, part, blocks, rows):
        return (self._expand(part, blocks, rows)[:, part.keep] - part.mean) / part.std

    def fill(self, blocks, rows, out):
        """Write the design of ``rows`` into ``out``, one row block per thread; the work is NumPy that releases the GIL.

        Every entry is computed from its own row, so the result does not depend on the row blocks.
        """
        def fill_rows(start):
            sub = slice(rows.start + start, min(rows.start + start + _THREAD_ROWS, rows.stop))
            Z = out[start:start + sub.stop - sub.start]
            Z[:, 0] = 1.
            for part in self.parts:
                Z[:, part.beta] = self.part_matrix(part, blocks, sub)

        with ThreadPoolExecutor(_THREADS) as pool:
            list(pool.map(fill_rows, range(0, rows.stop - rows.start, _THREAD_ROWS)))
        return out

    def matrix(self, blocks, rows, dtype=np.float64):
        return self.fill(blocks, rows, np.empty((rows.stop - rows.start, self.n_columns), dtype=dtype))

    def cache(self, blocks, n):
        """Hold the design in float32 so Newton passes do not rebuild it."""
        self.cached = self.fill(blocks, slice(0, n), np.empty((n, self.n_columns), dtype=np.float32))

    def rows(self, blocks, rows):
        return self.cached[rows] if self.cached is not None else self.matrix(blocks, rows)


class NewtonSolver:
    """Joint Newton (IRLS) for log forms: all blocks at once, on an identifiable standardized design.

    Each pass accumulates the log-likelihood gradient (float64) and Hessian (float32 when the design
    is cached) over row chunks; a step that raises the objective is halved. If halving fails with a
    float32 Hessian, the last point is re-evaluated with a float64 Hessian, which is kept from then on
    (``convergence_info_['hessian']`` records it). Frozen submodels enter as
    an offset ``log(prediction)`` and must predict positive values. After the fit, coefficients are
    mapped back to each block's own basis and normalized like a BCD fit: blocks with a constant
    direction have weighted mean 1, interaction curves have mean 1 within each level (the level
    moves to the ``by`` block), and ``global_scalar_`` carries the intercept and the means. Earlier
    steps are normalized for the history with block means from one extra pass over the design. Up
    to 8 threads over row blocks assemble the design and sum those means.

    ``solver_options``: ``max_iter`` (Newton steps), ``tol`` (relative objective decrease that stops),
    ``l2`` (ridge on the standardized coefficients, intercept excluded; a small value keeps levels
    without events finite), ``chunk_rows``, ``design_cache`` (``"auto"``, ``True`` or ``False``) and
    ``max_cache_gb`` (the float32 design is cached under ``"auto"`` when it fits).
    """

    name = "newton"
    FIT_ARGUMENTS = ("solver_options",)
    OPTIONS = {"max_iter": 50, "tol": 1e-10, "l2": 0., "chunk_rows": 500_000,
               "design_cache": "auto", "max_cache_gb": 16.}

    def __init__(self, model, form=None, *, solver_options=None, verbose=1):
        form = form if form is not None else model.form_
        if not hasattr(form, "newton_terms"):
            raise ValueError(f"Newton needs a form with newton_terms; got {form.name}")
        options = dict(solver_options or {})
        unknown = sorted(set(options) - set(self.OPTIONS))
        if unknown:
            raise ValueError(f"Unknown solver_options {unknown}; the Newton solver takes {sorted(self.OPTIONS)}")
        if options.get("design_cache", "auto") not in ("auto", True, False):
            raise ValueError("design_cache must be 'auto', True or False")
        self.model, self.form, self.verbose = model, form, verbose
        self.options = {**self.OPTIONS, **options}

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self, X, data_blocks, y, weights, started):
        """Fit ``self.model`` on container ``X``; ``started`` is when ``fit`` began, for ``setup_seconds``."""
        model, options = self.model, self.options
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
        design = _LogDesign(model, data_blocks, X.orig, w, options["chunk_rows"])
        offset = self._offset(design, data_blocks, len(y))
        self._warn_levels_without_events(design, y, w)
        size_gb = len(y) * design.n_columns * 4 / 1e9
        cache = options["design_cache"] if options["design_cache"] != "auto" else size_gb <= options["max_cache_gb"]
        if cache:
            design.cache(data_blocks, len(y))
        model.fit_timing_ = {'setup_seconds': time.perf_counter() - started, 'block_seconds': {},
                             'sweep_seconds': [], 'total_seconds': 0.,
                             'design_cache': 'float32' if cache else 'stream', 'design_columns': design.n_columns}
        if self.verbose > 0:
            mode = "cached float32 design" if cache else "design rebuilt each pass"
            print(f"Newton: {design.n_columns} design columns, {mode} ({size_gb:.1f} GB)")

        accepted, status, steps = self._iterate(design, data_blocks, y, w, offset)
        # Earlier steps only feed the history, so their block means come from one pass over the design;
        # the fitted step is normalized on the blocks themselves.
        means = self._factor_means(design, data_blocks, w, [beta for beta, *_ in accepted[:-1]]) + [None]
        design.cached = None

        history = [self._block_coefficients(design, beta, data_blocks, w, step_means)
                   for (beta, *_), step_means in zip(accepted, means)]
        params, curves, scalar = history[-1]
        model.coef_history_ = [h[0] for h in history]
        model.interaction_params_history_ = [h[1] for h in history]
        model.global_scalar_history_ = [h[2] for h in history]
        model.global_scalar_step_history_ = [(step, 'newton', h[2]) for step, h in enumerate(history)]
        model.loss_history_ = [loss for *_, loss in accepted]
        model.best_loss_, model.best_iteration_ = model.loss_history_[-1], len(accepted) - 1
        model.best_params_, model.best_interaction_params_ = params, curves
        model.coef_ = {group: InteractionCoef(by=model.interactions_[group], categories=curves[group])
                       if group in curves else params[group] for group in model.feature_groups_}
        model.global_scalar_ = scalar
        model.convergence_info_ = {'reason': status, 'iteration': steps, 'current_loss': model.best_loss_,
                                   'best_iteration': len(accepted), 'best_loss': model.best_loss_,
                                   'deviance': accepted[-1][1], 'solver': self.name,
                                   'hessian': self.hessian_precision,
                                   'pivots': {part.group: {'column': model.feature_groups_[part.group][part.pivot],
                                                           'from': 'reference' if part.pivot_from_reference else 'auto'}
                                              for part in design.parts if part.pivot is not None}}
        if self.verbose > 0:
            print(f"Newton stopped ({status}) after {steps} steps; Poisson deviance {accepted[-1][1]:.6e}.")
        model._compute_block_means(data_blocks, X, weights)
        model.fit_timing_['total_seconds'] = time.perf_counter() - started

    # ------------------------------------------------------------------
    # pieces
    # ------------------------------------------------------------------

    def _offset(self, design, blocks, n):
        """``log`` of the frozen submodels' product, or ``None`` without frozen blocks."""
        if not design.frozen:
            return None
        offset = np.zeros(n)
        for group in design.frozen:
            prediction = np.asarray(self.model.submodels_[group].predict(blocks[group]), dtype=float)
            if not (np.all(np.isfinite(prediction)) and np.all(prediction > 0)):
                raise ValueError(f"Frozen submodel '{group}' must predict finite positive values for a log form")
            offset += np.log(prediction)
        return offset

    def _warn_levels_without_events(self, design, y, w):
        if self.options["l2"]:
            return
        empty = []
        for part in design.parts:
            if part.kind == "categorical":
                codes = design.codes[part.group]
                weight = np.bincount(codes, weights=w, minlength=part.n_block_columns)
                events = np.bincount(codes, weights=w * y, minlength=part.n_block_columns)
                empty += [f"{part.group}[{level}]" for level in np.flatnonzero((weight > 0) & (events == 0))]
        if empty:
            warnings.warn("Levels without events have a maximum-likelihood factor of 0 and slow Newton down: "
                          f"{empty}; solver_options={{'l2': 1e-8}} keeps them finite.", stacklevel=4)

    def _iterate(self, design, blocks, y, w, offset):
        """Newton steps with step halving; returns the accepted ``(beta, deviance, objective, loss)`` and status."""
        model, form = self.model, self.form
        max_iter, tol, l2 = self.options["max_iter"], self.options["tol"], self.options["l2"]
        n, p = len(y), design.n_columns
        w_total = w.sum()
        beta = np.zeros(p)
        level = np.average(y, weights=w)
        if offset is not None:
            level /= np.average(np.exp(offset), weights=w)
        beta[0] = np.log(level)
        penalty = np.full(p, l2 * w_total)       # l2 on the scale of the summed log-likelihood
        penalty[0] = 0.
        accepted, step, t, steps, status = [], None, 1., 0, "max_iter"
        # A cached design gives a float32 Hessian; it switches to float64 if it fails to give a descent step.
        precise, reevaluate = design.cached is None, False
        self.hessian_precision = "float64" if precise else "float32"
        while True:
            pass_started = time.perf_counter()
            deviance = loss = 0.
            gradient, hessian = np.zeros(p), np.zeros((p, p))
            for rows in design.chunks(n):
                Z = design.rows(blocks, rows)
                # Convert a float32 chunk once and reuse it: NumPy would otherwise upcast it for each product.
                Z64 = Z.astype(np.float64, copy=False)
                eta = Z64 @ beta
                if offset is not None:
                    eta += offset[rows]
                mu, dev, grad_w, hess_w = form.newton_terms(eta, y[rows], w[rows])
                deviance += dev.sum()
                loss += np.dot(w[rows], form.loss_terms(y[rows], mu))
                gradient += grad_w @ Z64                             # float64: sets the optimum
                Zh = Z64 if precise else Z                           # Hessian only steers the step
                Zs = Zh * np.sqrt(hess_w).astype(Zh.dtype)[:, None]
                hessian += Zs.T @ Zs
            model.fit_timing_['sweep_seconds'].append(time.perf_counter() - pass_started)
            deviance, loss = deviance / w_total, loss / w_total
            objective = deviance + l2 * (beta[1:] @ beta[1:])
            if reevaluate:                                           # same point, now with a float64 Hessian
                reevaluate = False
            elif accepted and objective > accepted[-1][2]:
                if t >= 2 ** -10:
                    t /= 2                                           # overshoot: halve the last step
                    beta = accepted[-1][0] + t * step
                    continue
                if precise:
                    status = "line_search"
                    break
                precise, reevaluate = True, True
                self.hessian_precision = "float32, then float64"
                beta = accepted[-1][0].copy()
                continue
            else:
                accepted.append((beta.copy(), deviance, objective, loss))
                if self.verbose > 0:
                    print(f"Newton step {steps}: deviance {deviance:.6e}" + (f", step size {t:g}" if steps else ""))
                if len(accepted) > 1 and accepted[-2][2] - objective <= tol * accepted[-2][2]:
                    status = "converged"
                    break
                if steps == max_iter:
                    break
            system = hessian + np.diag(penalty)
            system[np.diag_indices(p)] += 1e-10 * np.trace(hessian) / p
            rhs = gradient - penalty * beta
            try:
                step = cho_solve(cho_factor(system), rhs)
            except np.linalg.LinAlgError:                           # singular, e.g. levels without events
                step = np.linalg.lstsq(system, rhs, rcond=None)[0]
            t, steps = 1., steps + 1
            beta = accepted[-1][0] + step
        return accepted, status, steps

    def _factor_means(self, design, blocks, w, betas):
        """Weighted means of the block factors at each of ``betas``, from one pass over the design.

        Returns one ``{group: mean}`` per beta; an interaction's mean is an array over its ``by``
        block's columns. A block's factor is ``factor`` of its own part of the design, which equals
        the factor of the mapped coefficients up to the design's float32 rounding when cached.
        """
        parts = [part for part in design.parts if part.kind != "categorical" and part.constant is not None]
        if not betas or not parts:
            return [{} for _ in betas]
        B = np.column_stack(betas)

        def chunk_sums(start):
            rows = slice(start, min(start + _THREAD_ROWS, len(w)))
            Z = design.rows(blocks, rows).astype(np.float64, copy=False)
            sums = {}
            for part in parts:
                weighted = w[rows, None] * self.form.factor(Z[:, part.beta], B[part.beta])
                if part.kind == "interaction":                       # levels x betas
                    codes, n_levels = design.codes[part.by][rows], blocks[part.by].shape[1]
                    sums[part.group] = np.column_stack([np.bincount(codes, weights=column, minlength=n_levels)
                                                        for column in weighted.T])
                else:
                    sums[part.group] = weighted.sum(axis=0)
            return sums

        sums = {part.group: 0. for part in parts}
        with ThreadPoolExecutor(_THREADS) as pool:                  # summed in row order: deterministic
            for chunk in pool.map(chunk_sums, range(0, len(w), _THREAD_ROWS)):
                for group, value in chunk.items():
                    sums[group] = sums[group] + value
        totals = {part.by: np.bincount(design.codes[part.by], weights=w, minlength=blocks[part.by].shape[1])
                  for part in parts if part.kind == "interaction"}
        means = {part.group: sums[part.group] / (totals[part.by][:, None] if part.kind == "interaction" else w.sum())
                 for part in parts}
        return [{group: mean[..., step] for group, mean in means.items()} for step in range(len(betas))]

    def _block_coefficients(self, design, beta, blocks, w, means=None):
        """Map design coefficients to each block's basis, normalized like a BCD fit.

        Returns ``(params, curves, global_scalar)``: coefficient vectors for regular blocks (NaN
        placeholders for frozen ones), ``{group: {label: coef}}`` for interaction curves, and the
        scalar carrying the intercept and every block mean. The means are evaluated on the blocks
        unless ``means`` (from ``_factor_means``) gives them.
        """
        form = self.form
        log_scalar = float(beta[0])
        params = {group: np.full(blocks[group].shape[1], np.nan) for group in design.frozen}
        curves, level_shifts, linear = {}, {}, {}
        for part in design.parts:
            slope = np.zeros(len(part.columns) * (blocks[part.by].shape[1] if part.by else 1))
            slope[part.keep] = beta[part.beta] / part.std
            linear[part.group] = (slope, -float(beta[part.beta] @ (part.mean / part.std)))

        for part in design.parts:
            if part.kind == "categorical":
                continue
            slope, intercept = linear[part.group]
            X = blocks[part.group]
            constant = part.constant
            if constant is None:
                log_scalar += intercept
            if part.kind == "numeric":
                coef = np.zeros(part.n_block_columns)
                coef[part.columns] = slope
                if constant is not None:
                    coef += intercept * constant
                    mean = means[part.group] if means is not None else np.average(form.factor(X, coef), weights=w)
                    coef = form.rescale(coef, mean, constant)
                    log_scalar += np.log(mean)
                params[part.group] = coef
                continue
            slopes = slope.reshape(len(part.columns), -1)
            shifts = np.zeros(slopes.shape[1])
            curves[part.group] = {}
            for label, column in design.levels[part.group].items():
                coef = np.zeros(part.n_block_columns)
                coef[part.columns] = slopes[:, column]
                if constant is not None:
                    coef += intercept * constant
                    if means is not None:
                        mean = means[part.group][column]
                    else:
                        rows = design.codes[part.by] == column
                        mean = np.average(form.factor(X[rows], coef), weights=w[rows])
                    coef = form.rescale(coef, mean, constant)
                    shifts[column] = np.log(mean)
                curves[part.group][label] = coef
            level_shifts[part.by] = level_shifts.get(part.by, 0.) + shifts

        for part in design.parts:
            if part.kind != "categorical":
                continue
            slope, intercept = linear[part.group]
            coef = slope + intercept + level_shifts.get(part.group, 0.)
            share = np.bincount(design.codes[part.group], weights=w, minlength=part.n_block_columns) / w.sum()
            mean = float(share @ form.factor(np.eye(part.n_block_columns), coef))
            params[part.group] = form.rescale(coef, mean, np.ones(part.n_block_columns))
            log_scalar += np.log(mean)
        return params, curves, float(np.exp(log_scalar))
