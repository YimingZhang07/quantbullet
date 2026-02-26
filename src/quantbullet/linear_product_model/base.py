import functools
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from .utils import (
    init_betas_by_response_mean, 
    estimate_ols_beta_se_with_scalar_vector
)
from .datacontainer import ProductModelDataContainer
from quantbullet.log_config import setup_logger
from dataclasses import dataclass
logger = setup_logger(__name__)

@dataclass
class FitMetadata:
    n_obs              : int
    n_features         : int
    offset_y           : float | None
    mean_y             : float
    ftol               : float | None = None
    cache_qr_decomp    : bool  | None = None

@dataclass
class InteractionCoef:
    """Per-category coefficients for a continuous-by-categorical interaction.

    Each entry in ``categories`` is either an ndarray (OLS-fitted coefficients
    sharing the same basis/knots) or an object with a ``.predict()`` method
    (a frozen submodel whose predictions are held constant during BCD).

    ``scalars`` holds per-category level corrections (A/E ratios) computed
    after BCD convergence to ensure each category's predictions are unbiased.
    The full prediction for category *c* is ``scalars[c] * X @ categories[c]``.
    """
    by: str
    categories: dict         # {cat_val: ndarray | model_with_predict}
    scalars: dict = None     # {cat_val: float}

    def __post_init__(self):
        if self.scalars is None:
            self.scalars = {k: 1.0 for k in self.categories}

def memorize_fit_args(func):
    """Decorator for fit methods to memorize some additional info about the input data X and y."""
    @functools.wraps(func)
    def wrapper( self, X, *args, **kwargs ):
        self.fit_metadata_ = FitMetadata(
            n_obs       = X.shape[0],
            n_features  = X.shape[1],
            offset_y    = kwargs.get('offset_y', None),
            mean_y      = np.mean( X.response ),
            ftol        = kwargs.get('ftol', None),
            cache_qr_decomp = kwargs.get('cache_qr_decomp', False)
        )
        return func( self, X, *args, **kwargs )
    return wrapper

class LinearProductModelBCD(ABC):
    """Mixin for models fitted via Block Coordinate Descent (BCD).

    Tracks optimisation history and provides normalised coefficient views.

    BCD-specific coefficient views (all derived from ``coef_``):

    - ``bias_one_coef_`` : each group's coefficients divided by its first
      element (the intercept/bias term), so the first coefficient is always 1.
    - ``normalized_coef_`` : each group's coefficients divided by its
      training-data block mean, making groups comparable in scale.
    - ``bias_one_coef_dict`` / ``normalized_coef_dict`` : named-dict versions
      of the above (``{group: {feature_name: value}}``).
    """

    def __init__(self):
        self._reset_history()

    def _reset_history( self, cache_qr_decomp=False ):
        self.coef_history_ = []
        self.coef_ = None
        self.loss_history_ = []
        self.best_loss_ = float('inf')
        self.best_params_ = None
        self.best_interaction_params_ = None
        self.best_iteration_ = None
        self.global_scalar_ = 1.0
        self.global_scalar_history_ = []
        self.interaction_params_history_ = []
        self.block_means_ = {}
        if cache_qr_decomp:
            self.qr_decomp_cache_ = {}

    @abstractmethod
    def loss_function(self, y_hat, y):
        pass
    
    @property
    def bias_one_coef_dict(self):
        """Return bias-one normalized coefficients grouped by feature group."""
        return self._coef_to_coef_dict(self.bias_one_coef_)
    
    @property
    def normalized_coef_dict(self):
        """Return block-mean normalized coefficients grouped by feature group."""
        return self._coef_to_coef_dict(self.normalized_coef_)
    
    def _coef_to_coef_dict(self, coef):
        """Convert coef_ dict to a nested dictionary with feature names.

        Interaction groups (InteractionCoef) are skipped -- they don't have a
        single coefficient vector that maps 1-to-1 to expanded feature names.
        """
        if self.feature_groups_ is None:
            raise ValueError("feature_groups_ is not set.")
        coef_dict = {}
        for group, features in self.feature_groups_.items():
            group_coef = coef.get(group)
            if group_coef is None or isinstance(group_coef, InteractionCoef):
                continue
            coef_dict[group] = {features[i]: group_coef[i] for i in range(len(features))}
        return coef_dict
    
    @property
    def bias_one_coef_(self):
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")
        bias_one_coef = {}
        for group, coef in self.coef_.items():
            if isinstance(coef, InteractionCoef):
                continue
            bias_one_coef[group] = coef / coef[0]
        return bias_one_coef
    
    @property
    def normalized_coef_(self):
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")
        normalized_coef = {}
        for group, coef in self.coef_.items():
            if isinstance(coef, InteractionCoef):
                continue
            block_mean = self.block_means_.get(group)
            normalized_coef[group] = coef / block_mean
        return normalized_coef


class LinearProductModelBase(ABC):
    """Core base class for all linear product models.

    A linear product model predicts
    ``y_hat = global_scalar * prod_i( X_i @ beta_i )``
    where each *feature group* ``i`` contributes one multiplicative factor.

    Attributes (set after ``fit()``)
    --------------------------------
    feature_groups_ : dict[str, list[str]]
        ``{group_name: [expanded_feature_names]}``.  Defines which columns
        of the expanded design matrix belong to each group.
    coef_ : dict[str, np.ndarray]
        Fitted coefficients.  ``{group_name: coefficient_vector}``.
        This is the canonical representation -- all other views derive from it.
    global_scalar_ : float
        A single multiplicative scalar applied to the entire product.
    se_ : dict[str, np.ndarray] | None
        Standard errors per group (populated by ``calculate_se``).

    Coefficient views (properties)
    ------------------------------
    coef_vector : np.ndarray
        All coefficients flattened into one array (for scipy optimisers).
    coef_dict : dict[str, dict[str, float]]
        Named version: ``{group: {feature_name: value}}``.
    """
    def __init__(self):
        self.feature_groups_ = None
        self.se_ = None
        self.coef_ = None
        self.global_scalar_ = 1.0

    def flatten_params(self, params_blocks):
        """Flatten the parameters from a dictionary of blocks into a single array."""
        if not isinstance(params_blocks, dict):
            raise ValueError("params_blocks must be a dictionary.")
        
        flat_params = []
        for key in self.block_names:
            if key not in params_blocks:
                raise ValueError(f"Feature group '{key}' not found in params_blocks.")
            flat_params.extend(params_blocks[key])
        
        return np.array(flat_params, dtype=float)
    
    def unflatten_params(self, flat_params):
        """Unflatten the parameters from a single array into a dictionary of blocks."""
        if not isinstance(flat_params, np.ndarray):
            raise ValueError("flat_params must be a numpy array.")
        
        params_blocks = {}
        start = 0
        for key in self.block_names:
            if key not in self.feature_groups_:
                raise ValueError(f"Feature group '{key}' not found in feature_groups_.")
            n_features = len(self.feature_groups_[key])
            params_blocks[key] = flat_params[start: start + n_features]
            start += n_features
        
        return params_blocks
    
    @property
    def coef_vector(self):
        """Return the coefficients as a flattened numpy array."""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")
        return self.flatten_params(self.coef_)
    
    @property
    def coef_dict(self):
        """Return coefficients as ``{group: {feature_name: value}}``.

        Interaction groups are skipped (use ``coef_[group].categories`` to
        inspect per-category coefficients).
        """
        if self.feature_groups_ is None:
            raise ValueError("feature_groups_ is not set.")
        result = {}
        for group in self.feature_groups_:
            coef = self.coef_.get(group)
            if coef is None or isinstance(coef, InteractionCoef):
                continue
            result[group] = dict(zip(self.feature_groups_[group], coef))
        return result
    
    def _coef_dict_to_blocks(self, coef_dict):
        """Convert a dictionary of coefficients to a dictionary of blocks.
        
        Returns
        -------
        dict
            { feature_group_name : <feature_block_array> }
        """
        if not isinstance(coef_dict, dict):
            raise ValueError("coef_dict must be a dictionary.")
        
        blocks = {}
        for key, features in coef_dict.items():
            if key not in self.feature_groups_:
                raise ValueError(f"Feature group '{key}' not found in feature_groups_.")
            blocks[key] = np.array([features[feature] for feature in self.feature_groups_[key]], dtype=float)
        
        return blocks

    @property
    def n_features(self):
        if self.feature_groups_ is None:
            raise ValueError("feature_groups_ is not set.")
        return sum(len(cols) for cols in self.feature_groups_.values())

    @property
    def block_names(self):
        """Feature group names, in the order used by flatten/unflatten."""
        if self.feature_groups_ is None:
            raise ValueError("feature_groups_ is not set.")
        return list(self.feature_groups_.keys())

    def infer_init_params(self, init_params, X_blocks, y):
        """Infer initial parameter guess from ``X_blocks`` structure.

        All structural information (block names, per-block dimensions) is
        derived from ``X_blocks`` directly, so this method has **no hidden
        dependence** on ``self.feature_groups_``.

        Parameters
        ----------
        init_params : None, scalar, or np.ndarray
            ``None`` → initialise so each block predicts ``mean(y)^(1/n_blocks)``.
            Scalar  → fill every coefficient with that value.
            Array   → use as-is (length must match total features).
        X_blocks : dict[str, np.ndarray]
            ``{group_name: design_matrix}``.  Keys define block ordering;
            ``design_matrix.shape[1]`` defines per-block dimensionality.
        y : np.ndarray
            Response variable (used only when ``init_params is None``).

        Returns
        -------
        flat_params : np.ndarray
            All block coefficients concatenated in key order.
        params_blocks : dict[str, np.ndarray]
            ``{group_name: coefficient_vector}`` (same data, dict form).
        """
        block_names = list(X_blocks.keys())
        block_sizes = {key: X_blocks[key].shape[1] for key in block_names}
        n_features = sum(block_sizes.values())

        if init_params is None:
            true_mean = np.mean(y)
            block_target = true_mean ** (1 / len(block_names))
            params_blocks = {
                key: init_betas_by_response_mean(X_blocks[key], block_target)
                for key in block_names
            }
            logger.debug(f"Using initial params: {params_blocks}")
        elif np.isscalar(init_params):
            params_blocks = {
                key: np.full(block_sizes[key], float(init_params), dtype=float)
                for key in block_names
            }
        elif isinstance(init_params, np.ndarray):
            if len(init_params) != n_features:
                raise ValueError(
                    f"init_params length {len(init_params)} does not match "
                    f"number of features {n_features}.")
            init_params = np.asarray(init_params, dtype=float)
            params_blocks, start = {}, 0
            for key in block_names:
                params_blocks[key] = init_params[start: start + block_sizes[key]]
                start += block_sizes[key]
        else:
            raise ValueError("init_params must be None, a numpy array, or a scalar.")

        flat_params = np.concatenate([params_blocks[key] for key in block_names])
        return flat_params, params_blocks

    def leave_out_feature_group_predict( self, group_to_exclude, X : ProductModelDataContainer | pd.DataFrame, params_dict = None, ignore_global_scale=False ):
        """Predict the product of all other feature groups except the one specified."""
        if not group_to_exclude in self.feature_groups_:
            raise ValueError(f"Feature group '{group_to_exclude}' not found in feature_groups_.")

        if params_dict is None:
            params_dict = self.coef_dict

        keep_feature_groups = { key: self.feature_groups_[key] for key in self.feature_groups_ if key != group_to_exclude }
        keep_params_dict = { key: params_dict[key] for key in params_dict if key != group_to_exclude }

        # Below is a fast path if the group to exclude is the only group
        if not keep_feature_groups or not keep_params_dict:
            if hasattr(self, 'global_scalar_'):
                return np.full(X.shape[0], self.global_scalar_)
            else:
                return np.ones(X.shape[0], dtype=float)

        if isinstance(X, ProductModelDataContainer):
            X_blocks = X.get_containers_dict( list( keep_feature_groups.keys() ) )
        elif isinstance(X, pd.DataFrame):
            X_blocks = { key: X[cols].values for key, cols in keep_feature_groups.items() }
        else:
            raise ValueError("Invalid input type. Expected ProductModelDataContainer or pd.DataFrame.")

        params_blocks = self._coef_dict_to_blocks( keep_params_dict )

        # params blocks are only useful for the blocks with linear terms. the submodels will use their own parameters
        preds = self.forward( params_blocks, X_blocks, ignore_global_scale=ignore_global_scale )
        return preds
    
    def single_feature_group_predict(self, group_to_include, X, params_dict = None, ignore_global_scale=True):
        """Predict only the specified feature group. Note that this excludes the global scalar"""
        if not group_to_include in self.feature_groups_:
            raise ValueError(f"Feature group '{group_to_include}' not found in feature_groups_.")
        
        if params_dict is None:
            params_dict = self.coef_dict

        include_feature_groups = { group_to_include: self.feature_groups_[group_to_include] }
        include_params_dict = { group_to_include: params_dict[group_to_include] }

        X_blocks = X.get_expanded_array_dict( list( include_feature_groups.keys() ) )
        params_blocks = self._coef_dict_to_blocks( include_params_dict )

        preds = self.forward(params_blocks, X_blocks, ignore_global_scale=ignore_global_scale)
        return preds
    
    @abstractmethod
    def calculate_feature_group_se(self, feature_group, X, y):
        """
        Calculate the standard error of the coefficients for a specific feature group.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def calculate_se(self, X, y):
        """
        Calculate the standard error of the coefficients for all feature groups.
        This method iterates over all feature groups and calls calculate_feature_group_se.
        """
        se_blocks = {}
        for group in self.feature_groups_:
            se_blocks[group] = self.calculate_feature_group_se(group, X, y)
        
        self.se_ = se_blocks
        return se_blocks
    
    def summary(self):
        """Generate a summary of the model coefficients and standard errors, and confidence intervals."""
        # we can access coef_ and se_, they are dictionary of feature group names and array of coefficients or standard errors
        # generate a DataFrame with columns feature group name, feature name, coefficient, standard error, and confidence interval
        # each value should round to 4 decimal places
        if self.coef_ is None or self.se_ is None:
            raise ValueError("Model coefficients or standard errors are not available. Please fit the model first.")
        summary_data = []
        for group, features in self.feature_groups_.items():
            if group not in self.coef_ or group not in self.se_:
                continue
            for i, feature in enumerate(features):
                coef = self.coef_[group][i]
                se = self.se_[group][i]
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se
                summary_data.append({
                    'feature_group': group,
                    'feature_name': feature,
                    'coefficient': round(coef, 4),
                    'standard_error': round(se, 4),
                    'ci_lower': round(ci_lower, 4),
                    'ci_upper': round(ci_upper, 4)
                })
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
class LinearProductRegressorBase(LinearProductModelBase):
    def __init__(self):
        super().__init__()
        self.offset_y = None

    def _predict_group(self, group, X_block, cat_series=None):
        """Predict one group's multiplicative contribution.

        Dispatches based on the type stored in ``coef_[group]``:
        - ``np.ndarray``: standard linear ``X_block @ coef``
        - ``InteractionCoef``: per-category prediction assembled via masking
        - Submodel (in ``submodels_``): ``submodel.predict(X_block)``

        Parameters
        ----------
        group : str
        X_block : np.ndarray
            Design matrix for this group (n_obs, n_basis).
        cat_series : array-like, optional
            Categorical column values.  Required when ``coef_[group]`` is an
            ``InteractionCoef``.
        """
        coef = self.coef_[group]
        if isinstance(coef, InteractionCoef):
            if cat_series is None:
                raise ValueError(
                    f"cat_series is required for interaction group '{group}' "
                    f"(interaction by '{coef.by}')"
                )
            pred = np.ones(X_block.shape[0], dtype=float)
            for cat_val, cat_coef in coef.categories.items():
                mask = (cat_series == cat_val)
                if hasattr(mask, 'values'):
                    mask = mask.values
                scalar = coef.scalars.get(cat_val, 1.0)
                if hasattr(cat_coef, 'predict'):
                    pred[mask] = scalar * cat_coef.predict(X_block[mask])
                else:
                    pred[mask] = scalar * (X_block[mask] @ cat_coef)
            return pred
        if hasattr(self, 'submodels_') and group in self.submodels_:
            return self.submodels_[group].predict(X_block)
        return np.dot(X_block, coef)
        
    def calculate_feature_group_se( self, feature_group, X, y ):
        """Calculate the standard error of the coefficients for a specific feature group."""
        fixed_blocks_preds = self.leave_out_feature_group_predict(group_to_exclude=feature_group, X=X)
        X_block_coef = self.coef_[feature_group]
        X_block = X[self.feature_groups_[feature_group]].values
        return estimate_ols_beta_se_with_scalar_vector( X_block, y, beta=X_block_coef, scalar_vector=fixed_blocks_preds )
    
    def predict( self, X : ProductModelDataContainer | pd.DataFrame ):
        if self.feature_groups_ is None or self.coef_ is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")

        if isinstance(X, ProductModelDataContainer):
            data_blocks = X.get_expanded_array_dict( list( self.feature_groups_.keys() ) )
        elif isinstance(X, pd.DataFrame):
            data_blocks = { key: X[cols].values for key, cols in self.feature_groups_.items() }
        else:
            raise ValueError("Invalid input type. Expected ProductModelDataContainer or pd.DataFrame.")

        if self.offset_y is not None:
            # if there is an offset, we need to subtract it from the prediction
            return self.forward( self.coef_, data_blocks ) - self.offset_y
        return self.forward( self.coef_, data_blocks )
    
    def forward(self, params_blocks, X_blocks, ignore_global_scale=False):
        """
        Compute the forward pass for the model.

        Parameters
        ----------
        params_blocks : dict
            A dictionary mapping feature block names to their parameter vectors.
        X_blocks : dict
            A dictionary mapping feature block names to the data containers.

        Returns
        -------
        np.ndarray
            The model's predictions for the input data.
        """
        # Find any block to get n_obs
        n_obs = next(iter(X_blocks.values())).shape[0]

        result = np.ones(n_obs, dtype=float)
        for key in params_blocks:
            if key not in X_blocks:
                raise ValueError(f"Feature group '{key}' not found in input blocks.")

            # if the feature group is a submodel, use the submodel to predict
            if hasattr( self, 'submodels_' ) and key in self.submodels_:
                result *= self.submodels_[ key ].predict( X_blocks[ key ] )
            else:
                result *= np.dot( X_blocks[ key ], params_blocks[ key ] )

        if not ignore_global_scale:
            result = result * self.global_scalar_
        return result
    
class LinearProductClassifierBase(LinearProductModelBase):
    def __init__(self):
        super().__init__()
    
    def calculate_feature_group_se(self, feature_group, X, y):
        pass