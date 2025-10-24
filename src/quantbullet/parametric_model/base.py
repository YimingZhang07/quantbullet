import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod

class ParametricModel(ABC):
    __slots__ = ['params_dict', 'allow_extrapolation', 'left_bound_', 'right_bound_', '_model_name']

    # Default model name; subclasses should override
    default_model_name = "ParametricModel"

    def __init__(self, params_dict=None, allow_extrapolation=False, model_name=None):
        """
        Parameters
        ----------
        func : callable
            The function to fit.
        params : dict
            The parameters of the function.
        model_name : str | None
            Optional custom model name. If None, falls back to class default.
        """
        self.params_dict = params_dict
        self.allow_extrapolation = allow_extrapolation
        self.left_bound_ = None
        self.right_bound_ = None
        # Allow instance-level override of model name; None means use default
        self._model_name = model_name
        
    def fit(self, x, y, p0=None, bounds=(-np.inf, np.inf), weights=None, left_bound=None, right_bound=None):
        
        # NOTE the weights here correspond to the sigma parameter in curve_fit, which is not about the importance weights but the standard deviation
        self.left_bound_ = left_bound if left_bound is not None else np.min(x)
        self.right_bound_ = right_bound if right_bound is not None else np.max(x)
        param_names = self.get_param_names()
        popt, pcov = curve_fit(self.func_with_args, x, y, p0=p0, bounds=bounds, maxfev=10000, sigma=weights)
        # Convert fitted parameters back to dictionary format
        self.params_dict = dict(zip(param_names, popt))
        return self

    def predict(self, x):
        if not self.params_dict:
            raise ValueError("Model not fitted yet. Please call fit() first.")
        if not self.allow_extrapolation and self.left_bound_ is not None and self.right_bound_ is not None:
            x = np.clip(x, self.left_bound_, self.right_bound_)
        preds = self.func_with_kwargs(x, **self.params_dict)
        preds = np.asarray( preds ).ravel()
        return preds

    @property
    def model_name(self):
        return self._model_name if self._model_name is not None else self.default_model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @abstractmethod
    def func_with_kwargs( self, x, **kwargs ):
        pass

    @abstractmethod
    def func_with_args( self, x, *args ):
        pass

    @abstractmethod
    def get_param_names(self):
        """
        Return list of parameter names in the order expected by func_for_fitting.
        """
        pass

    @classmethod
    def from_dict(cls, data_dict):
        # Extract constructor parameters
        constructor_params = {
            'params_dict': data_dict.get('params_dict'),
            'allow_extrapolation': data_dict.get('allow_extrapolation', False),
            'model_name': data_dict.get('_model_name', data_dict.get('model_name'))
        }
        
        # Create instance
        instance = cls(**constructor_params)
        
        # Set other attributes that aren't constructor parameters
        if 'left_bound_' in data_dict:
            instance.left_bound_ = data_dict['left_bound_']
        if 'right_bound_' in data_dict:
            instance.right_bound_ = data_dict['right_bound_']
        return instance

    def to_dict(self):
        # Use __slots__ instead of __dict__
        result = {}
        for slot in self.__slots__:
            if hasattr(self, slot):
                result[slot] = getattr(self, slot)
        if result['_model_name'] is None:
            result['_model_name'] = self.default_model_name
        return result

    def __repr__(self):
        return f"{self.model_name}({self.params_dict})"

    def plot_data_and_model(self, x, y, **kwargs):
        fig, ax = plt.subplots()
        ax.plot(x, y, **kwargs)
        ax.plot(x, self.predict(x), **kwargs)
        return fig, ax
    
    @abstractmethod
    def math_repr(self):
        pass