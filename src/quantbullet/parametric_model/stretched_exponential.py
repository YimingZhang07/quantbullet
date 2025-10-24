"""
Models a "decrease then plateau" relationship.
"""

from .base import ParametricModel
import numpy as np

class StretchedExponentialModel( ParametricModel ):
    default_model_name = "StretchedExponentialModel"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func_with_kwargs( self, x, **params_dict ):
        """
        Stretched Exponential decay to plateau.
        y(t)=a+b \exp \left[-\left(\frac{t}{\tau}\right)^k\right]

        Parameters
        ----------
        a : float
            Lower plateau (min CPR/SMM).
        b : float
            Scale parameter.
        tau : float
            Characteristic time.
        k : float
            Stretching exponent.
        """
        a = params_dict['a']
        b = params_dict['b']
        tau = params_dict['tau']
        k = params_dict['k']
        return a + b * np.exp( - (x / tau) ** k )

    def func_with_args(self, x, a, b, tau, k):
        return self.func_with_kwargs(x, a=a, b=b, tau=tau, k=k)

    def get_param_names(self):
        return ['a', 'b', 'tau', 'k']

    def math_repr( self ):
        return "f(x) = a + b * exp(- (x / tau) ** k)"