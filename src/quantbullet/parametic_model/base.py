import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod

class ParametricModel(ABC):
    __slots__ = ['params_dict', 'allow_extrapolation', 'left_bound_', 'right_bound_']
    def __init__(self, params_dict=None, allow_extrapolation=False):
        """
        Parameters
        ----------
        func : callable
            The function to fit.
        params : dict
            The parameters of the function.
        """
        self.params_dict = params_dict
        self.allow_extrapolation = allow_extrapolation
        self.left_bound_ = None
        self.right_bound_ = None
        
    def fit(self, x, y, p0=None, bounds=(-np.inf, np.inf), weights=None):

        self.left_bound_ = np.min(x)
        self.right_bound_ = np.max(x)
        param_names = self.get_param_names()
        popt, pcov = curve_fit(self.func_with_args, x, y, p0=p0, bounds=bounds, maxfev=10000, sigma=weights)
        # Convert fitted parameters back to dictionary format
        self.params_dict = dict(zip(param_names, popt))
        return self

    def predict(self, x):
        if not self.params_dict:
            raise ValueError("Model not fitted yet. Please call fit() first.")
        if not self.allow_extrapolation:
            x = np.clip(x, self.left_bound_, self.right_bound_)
        return self.func_with_kwargs(x, **self.params_dict)

    @property
    @abstractmethod
    def model_name(self):
        pass

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
            'allow_extrapolation': data_dict.get('allow_extrapolation', False)
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
        return result

    def __repr__(self):
        return f"{self.model_name}({self.params_dict})"

    def plot_data_and_model(self, x, y, **kwargs):
        fig, ax = plt.subplots()
        ax.plot(x, y, **kwargs)
        ax.plot(x, self.predict(x), **kwargs)
        return fig, ax