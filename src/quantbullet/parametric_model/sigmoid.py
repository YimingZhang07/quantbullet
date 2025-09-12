from .base import ParametricModel
import numpy as np

class SigmoidModel( ParametricModel ):
    default_model_name = "SigmoidModel"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func_with_kwargs( self, x, **params_dict ):
        """
        Logistic S-curve for prepayment incentive.
        y = L / (1 + exp(-k * (x - x0)))
        
        Parameters
        ----------
        L : float
            Upper plateau (max CPR/SMM).
        k : float
            Steepness of curve.
        x0 : float
            Midpoint incentive.
        """
        L = params_dict['L']
        k = params_dict['k']
        x0 = params_dict['x0']
        return L / (1 + np.exp(-k * (x - x0)))

    def func_with_args( self, x, L, k, x0 ):
        return self.func_with_kwargs(x, L=L, k=k, x0=x0)

    def get_param_names(self):
        return ['L', 'k', 'x0']
    
    def math_repr( self ):
        return "f(x) = L / (1 + exp(-k * (x - x0)))"