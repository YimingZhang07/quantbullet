from .base import ParametricModel
import numpy as np

class AsymQuadModel( ParametricModel ):
    # Default name for this subclass; can be overridden per instance via model_name
    default_model_name = "AsymQuadModel"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func_with_kwargs( self, x, **params_dict ):
        a = params_dict['a']
        b = params_dict['b']
        x0 = params_dict['x0']
        c = params_dict['c']
        return a * (x - x0)**2 + b * (x - x0) + c

    def func_with_args( self, x, a, b, x0, c ):
        return self.func_with_kwargs(x, a=a, b=b, x0=x0, c=c)

    def get_param_names(self):
        return ['a', 'b', 'x0', 'c']