"""
Models a "decrease then plateau" relationship.
"""

from .base import ParametricModel
import numpy as np

class ExpPlateauModel( ParametricModel ):
    default_model_name = "ExpPlateauModel"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func_with_kwargs( self, x, **params_dict ):
        """
        Exponential decay to plateau.
        y = L - (L - y0) * exp(-k * x ** 2)
        
        Parameters
        ----------
        L : float
            Lower plateau (min CPR/SMM).
        k : float
            Rate of decay.
        y0 : float
            Starting value (y when x=0).
        """
        L = params_dict['L']
        k = params_dict['k']
        y0 = params_dict['y0']
        return L - (L - y0) * np.exp(-k * x ** 2)
    
    def func_with_args(self, x, L, k, y0):
        return self.func_with_kwargs(x, L=L, k=k, y0=y0)
    
    def get_param_names(self):
        return ['L', 'k', 'y0']
    
    def math_repr( self ):
        return "f(x) = L - (L - y0) * exp(-k * x^2)"