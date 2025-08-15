from abc import ABC, abstractmethod
import numpy as np
from .utils import init_betas_by_response_mean

class LinearProductModelBCD(ABC):
    """
    Base class for linear product models using Block Coordinate Descent (BCD).
    """

    def __init__(self):
        self._reset_history()

    def _reset_history( self ):
        self.params_history_ = []
        self.coef_ = None
        self.loss_history_ = []
        self.best_loss_ = float('inf')
        self.best_params_ = None
        self.best_iteration_ = None
        self.global_scale_ = 1.0
        self.global_scale_history_ = []
        self.block_means_ = {}

    @abstractmethod
    def loss_function(self, y_hat, y):
        pass

    @property
    def coef_dict(self):
        """Return coefficients grouped by feature group."""
        return self._coef_to_coef_dict(self.coef_)
    
    @property
    def bias_one_coef_dict(self):
        """Return bias-one normalized coefficients grouped by feature group."""
        return self._coef_to_coef_dict(self.bias_one_coef_)
    
    @property
    def normalized_coef_dict(self):
        """Return block-mean normalized coefficients grouped by feature group."""
        return self._coef_to_coef_dict(self.normalized_coef_)
    
    def _coef_to_coef_dict(self, coef):
        """Convert coef_ dict to a nested dictionary with feature names."""
        if self.feature_groups_ is None:
            raise ValueError("feature_groups_ is not set.")
        coef_dict = {}
        for group, features in self.feature_groups_.items():
            coef_dict[group] = {features[i]: coef[group][i] for i in range(len(features))}
        return coef_dict
    
    @property
    def bias_one_coef_(self):
        # assume the first feature in each group is the bias term
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")
        bias_one_coef = {}
        for group, coef in self.coef_.items():
            bias_coef = coef[0]
            bias_one_coef[group] = coef / bias_coef
        return bias_one_coef
    
    @property
    def normalized_coef_(self):
        if self.coef_ is None:
            raise ValueError("Model not fitted yet. Please call fit() first.")
        normalized_coef = {}
        for group, coef in self.coef_.items():
            block_mean = self.block_means_.get(group)
            normalized_coef[group] = coef / block_mean
        return normalized_coef


class LinearProductModelBase(ABC):
    def __init__(self):
        self.feature_groups_ = None

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

    def infer_init_params(self, init_params, X_blocks, y):
        """
        Infer initial guess, depending on the type of init_params.
        If init_params is None, it initializes based on the mean of the response variable.
        If init_params is a scalar, it initializes all blocks with that value.

        Parameters
        ----------
        init_params : None, scalar, or np.ndarray
            Initial parameters for the model.
        X_blocks : dict
            Dictionary of feature blocks, where keys are block names and values are feature matrices.
        y : np.ndarray
            Response variable for the model.

        Returns
        -------
        init_params : np.ndarray
            Flattened initial parameters for the model.
        init_params_blocks : dict
            Dictionary of initial parameters for each block, where keys are block names and values are parameter arrays.
        """
        if init_params is None:
            # we cannot use 1s as initial parameters anymore, as this leads to >1 predicted values and clipped to 1 for all observations
            # making it impossible to optimize;
            # Therefore we initialize a constant value that on average predicts the true probability
            true_mean = np.mean(y)
            n_blocks = len(self.block_names)
            block_target = true_mean ** (1 / n_blocks)
            init_params_blocks = { key: init_betas_by_response_mean(X_blocks[key], block_target) for key in self.block_names }
            print(f"Using initial params: {init_params_blocks}")
            init_params = self.flatten_params(init_params_blocks)
        else:
            if np.isscalar(init_params):
                init_params_blocks = { key: np.full(len(self.feature_groups_[key]), float(init_params), dtype=float) for key in self.block_names }
                init_params = self.flatten_params(init_params_blocks)
            elif isinstance(init_params, np.ndarray):
                if len(init_params) != self.n_features:
                    raise ValueError(f"init_params length {len(init_params)} does not match number of features {self.n_features}.")
                else:
                    init_params = np.asarray(init_params, dtype=float)
                    init_params_blocks = self.unflatten_params(init_params)
            else:
                raise ValueError("init_params must be None, a numpy array, or a scalar.")
            
        return init_params, init_params_blocks
