import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from abc import ABC, abstractmethod

class ParametricModel(ABC):
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
    def from_dict(cls, params_dict):
        return cls(params_dict, allow_extrapolation=params_dict.get("allow_extrapolation", False))

    def to_dict(self):
        all_dict = self.params_dict.copy()
        all_dict.update({
            "left_bound_": self.left_bound_,
            "right_bound_": self.right_bound_,
        })
        return all_dict

    def __repr__(self):
        return f"{self.model_name}({self.params_dict})"

    def plot_data_and_model(self, x, y, **kwargs):
        fig, ax = plt.subplots()
        ax.plot(x, y, **kwargs)
        ax.plot(x, self.predict(x), **kwargs)
        return fig, ax