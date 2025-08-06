import numpy as np
from scipy.optimize import least_squares

class PiecewiseLogProductOptimizer:
    """
    A class to optimize a piecewise log product function.
    """
    def __init__(self, method = 'trf', ftol = 1e-8, xtol = 1e-8, gtol = 1e-8, min_log_value = 1e-10):
        self.method = method
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.min_log_value = min_log_value

    @staticmethod
    def minimize_target( params, X_blocks, y ):
        y_hat = PiecewiseLogProductOptimizer.model_output(params, X_blocks)
        return y_hat - y
    
    @staticmethod
    def model_output( params, X_blocks, min_log_value=1e-10 ):
        """
        Calculate the model output for a piecewise log product function.
        
        Parameters
        ----------
        params : np.ndarray
            The coefficients for the model, should be a 1D numpy array.
        X_blocks : list of np.ndarray
            List of feature blocks, where each block is a 2D numpy array.
        min_inner_value : float, optional
            Minimum value for the inner product to avoid log(0) or negative values.
            Default is 1e-10.
        Returns
        -------
        np.ndarray
            The model output as a 1D numpy array.
        """
        
        if len(params) != sum(X.shape[1] for X in X_blocks):
            raise ValueError("The length of params must match the total number of features in X_blocks.")
        
        n_obs = X_blocks[0].shape[0]
        y_hat = np.zeros(n_obs)
        idx = 0
        for X in X_blocks:
            n_features = X.shape[1]
            theta = params[ idx:idx + n_features ]
            inner = X @ theta
            # Ensure inner product is not less than min_log_value to avoid log(0) or negative values
            inner = np.clip( inner, a_min=min_log_value )
            y_hat += np.log(inner)
            idx += n_features
        return y_hat
    
    @staticmethod
    def jacobian( params, X_blocks, min_inner_value=1e-10 ):
        """
        Calculate the Jacobian of the model output.
        
        Parameters
        ----------
        params : np.ndarray
            The coefficients for the model, should be a 1D numpy array.
        X_blocks : list of np.ndarray
            List of feature blocks, where each block is a 2D numpy array.
        min_inner_value : float, optional
            Minimum value for the inner product to avoid log(0) or negative values.
            Default is 1e-10.
        Returns
        -------
        np.ndarray
            The Jacobian as a 2D numpy array.
        """
        
        if len(params) != sum(X.shape[1] for X in X_blocks):
            raise ValueError("The length of params must match the total number of features in X_blocks.")
        
        n_obs = X_blocks[0].shape[0]
        n_features = sum(X.shape[1] for X in X_blocks)
        jacobian = np.zeros((n_obs, n_features))
        
        idx = 0
        for X in X_blocks:
            n_block_features = X.shape[1]
            theta = params[idx:idx + n_block_features]
            inner = X @ theta
            inner = np.clip(inner, a_min=min_inner_value)
            jacobian[:, idx:idx + n_block_features] = X / inner[:, None]
            idx += n_block_features
        
        return jacobian
    
    def _optimize( self, init_params, X_blocks, y ):
        result = least_squares(
            self.minimize_target,
            x0=init_params,
            args = ( X_blocks, y ),
            method=self.method,
            loss = 'linear',
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
            jac=self.jacobian
        )
        return result
    
    def optimize_with_X_blocks( self, X_blocks, y, init_params = None ):
        if not isinstance(X_blocks, list):
            raise ValueError("X_blocks must be a list data.")
        
        n_features = sum( block.shape[1] for block in X_blocks )
        
        if init_params is None:
            init_params = np.ones(n_features)

        return self._optimize( init_params, X_blocks, y )