from .base import ParametricModel
import numpy as np

class BathtubModel( ParametricModel ):
    default_model_name = "BathtubModel"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def func_with_kwargs( self, x, **params_dict ):
        """
        Bathtub curve using two Weibull-like terms.
        y = lam * ( (x/theta1)^k1 + (theta2/x)^k2 )
        """
        lam         = params_dict[ 'lam' ]
        theta1      = params_dict[ 'theta1' ]
        k1          = params_dict[ 'k1' ]
        theta2      = params_dict[ 'theta2' ]
        k2          = params_dict[ 'k2' ]
        x = np.asarray(x)
        x_safe = np.where(x == 0, 1e-8, x)
        return lam * ((x_safe / theta1)**k1 + (theta2 / x_safe)**k2)

    def func_with_args( self, x, lam, theta1, k1, theta2, k2 ):
        return self.func_with_kwargs(x, lam=lam, theta1=theta1, k1=k1, theta2=theta2, k2=k2)

    def get_param_names(self):
        """
        Return parameter names in the order expected by func_with_args.
        """
        return ['lam', 'theta1', 'k1', 'theta2', 'k2']