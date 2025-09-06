import numpy as np
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod

class ParametricModel(ABC):
    def __init__(self, params_dict=None):
        """
        Parameters
        ----------
        func : callable
            The function to fit.
        params : dict
            The parameters of the function.
        """
        self.params_dict = params_dict
        
    def fit(self, x, y, p0=None, bounds=(-np.inf, np.inf), weights=None):
        param_names = self.get_param_names()
        
        popt, pcov = curve_fit(self.func_with_args, x, y, p0=p0, bounds=bounds, maxfev=10000, sigma=weights)
        # Convert fitted parameters back to dictionary format
        self.params_dict = dict(zip(param_names, popt))
        return self

    def predict(self, x):
        if not self.params_dict:
            raise ValueError("Model not fitted yet. Please call fit() first.")
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
    def from_dict(cls, params_dict):
        return cls(params_dict)

    def to_dict(self):
        return self.params_dict
