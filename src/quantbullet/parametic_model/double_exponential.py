from .base import ParametricModel
import numpy as np

class DoubleExponentialModel( ParametricModel ):
    # Default name for this subclass; can be overridden per instance via model_name
    default_model_name = "DoubleExponentialModel"

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs )

    def func_with_kwargs( self, x, **params_dict ):
        alpha   = params_dict[ 'alpha' ]
        beta    = params_dict[ 'beta' ]
        gamma   = params_dict[ 'gamma' ]
        return alpha * ( 1 - np.exp( -beta * x ) ) * np.exp( -gamma * x)

    def func_with_args( self, x, alpha, beta, gamma ):
        return self.func_with_kwargs( x, alpha=alpha, beta=beta, gamma=gamma )

    def get_param_names(self):
        return [ 'alpha', 'beta', 'gamma' ]