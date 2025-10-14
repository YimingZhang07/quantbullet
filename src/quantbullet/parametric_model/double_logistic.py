"""
Hill Double Logistic Model is used to model "flat then increase then flat then decrease then flat" relationships.
"""

from .base import ParametricModel
import numpy as np

class DoubleLogisticModel( ParametricModel ):
    default_model_name = "DoubleLogisticModel"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func_with_kwargs( self, x, **params_dict ):
        """
        Double logistic function combining two sigmoid curves.
        
        Parameters:
        L1, L2: Upper plateaus of each sigmoid
        k1, k2: Steepness parameters
        x1, x2: Midpoint locations
        c: Baseline offset
        """
        L1 = params_dict['L1']
        L2 = params_dict['L2']
        k1 = params_dict['k1']
        x1 = params_dict['x1']
        k2 = params_dict['k2']
        x2 = params_dict['x2']
        c = params_dict['c']
        x = np.asarray(x)
        return (L1 / (1 + np.exp(-k1 * (x - x1))) +
                L2 / (1 + np.exp(-k2 * (x - x2))) + c)

    def func_with_args( self, x, L1, L2, k1, x1, k2, x2, c ):
        return self.func_with_kwargs(x, L1=L1, L2=L2, k1=k1, x1=x1, k2=k2, x2=x2, c=c)

    def get_param_names(self):
        return ['L1', 'L2', 'k1', 'x1', 'k2', 'x2', 'c']
    
    def math_repr( self ):
        return "f(x) = L1/(1 + exp(-k1*(x - x1))) + L2/(1 + exp(-k2*(x - x2))) + c"
    

class HillDoubleLogisticModel( ParametricModel ):
    default_model_name = "HillDoubleLogisticModel"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func_with_kwargs(self, x, **params_dict):
        A  = params_dict['A']
        k1 = params_dict['k1']
        x1 = params_dict['x1']
        k2 = params_dict['k2']
        x2 = params_dict['x2']
        c  = params_dict.get('c', 0)
        x = np.asarray(x)
        sig1 = 1 / (1 + np.exp(-k1 * (x - x1)))
        sig2 = 1 / (1 + np.exp(-k2 * (x - x2)))
        return A * (sig1 - sig2) + c

    def func_with_args(self, x, A, k1, x1, k2, x2, c=0):
        return self.func_with_kwargs(x, A=A, k1=k1, x1=x1, k2=k2, x2=x2, c=c)
    
    def get_param_names(self):
        return ['A', 'k1', 'x1', 'k2', 'x2', 'c']

    def math_repr(self):
        return "f(x) = A * (1/(1 + exp(-k1*(x - x1))) - 1/(1 + exp(-k2*(x - x2)))) + c"