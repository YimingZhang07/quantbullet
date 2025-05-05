import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from quantbullet.plot.colors import ColorEnum

class RollingRidgeImputer:
    """
    Rolling Exponential Weighted Ridge Regression.
    This class implements a rolling regression model using Ridge regression with exponential weighting.
    It is designed to handle time series data where the dependent variable (y) may have missing values.
    """
    def __init__(self, span=30, alpha=1.0):
        """
        Parameters
        ----------
        span : int
            The window size for rolling regression.
        alpha : float
            The regularization strength for Ridge regression.
        """
        self.span       = span
        self.alpha      = alpha
        self.coef_      = None
        self.fitted_    = None

        self._raw_y     = None
        self._raw_x     = None

    def fit(self, y: pd.Series, x: pd.Series):
        """
        Fit the model to sparse y and complete x using rolling EW Ridge.

        Parameters
        ----------
        y : pd.Series
            The dependent variable (target).
        x : pd.Series
            The independent variable (predictor).
        """
        assert isinstance( y, pd.Series ) and isinstance( x, pd.Series ), "Inputs must be pandas Series"
        assert y.index.equals(x.index), "x and y must be aligned"
        
        self._raw_y = y.copy()
        self._raw_x = x.copy()

        y_pred = pd.Series( index=y.index, dtype=float )
        coefs = []

        for t in range( self.span, len( y ) ):
            y_window = y.iloc[ t - self.span:t ]
            x_window = x.iloc[ t - self.span:t ]

            mask = y_window.notna()
            if mask.sum() < self.span // 2:
                coefs.append( ( np.nan, np.nan ) )
                continue

            # decay = 3
            # w = np.exp(-np.linspace(0, decay, mask.sum()))[::-1]
            w = np.exp( -np.linspace( 0, 1, mask.sum() ) )[ ::-1 ]
            model = Ridge( alpha=self.alpha, fit_intercept=True )
            model.fit( x_window[ mask ].values.reshape( -1, 1 ), y_window[ mask ].values, sample_weight=w )

            pred = model.predict( [ [ x.iloc[ t ] ] ] )[ 0 ]
            y_pred.iloc[t] = pred
            coefs.append( ( model.intercept_, model.coef_[ 0 ] ) )

        self.fitted_ = y_pred
        self.coef_ = pd.DataFrame( coefs, index=y.index[ self.span: ], columns=[ 'intercept', 'slope' ] )
        return self

    def get_fitted(self):
        """Return predicted y values (NaNs where not available)."""
        return self.fitted_

    def get_coefficients(self):
        """Return DataFrame of (intercept, slope) per day."""
        return self.coef_

    def plot(self):
        """Plot the fitted values and coefficients."""
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        # self.fitted_.plot(ax=axs[0], label='Fitted y', linewidth=2)

        self._raw_y.plot( ax=axs[0], label='y', color=ColorEnum.BLUE_L3.value, linewidth=2 )
        self._raw_x.plot( ax=axs[0], label='x', color=ColorEnum.GREEN_L3.value, alpha=0.5 )
        self.fitted_.plot( ax=axs[0], label='Predicted y', color=ColorEnum.ORANGE_L3.value, linewidth=2 )

        axs[0].legend(); axs[0].set_title( "Fitted y" )

        self.coef_[['intercept', 'slope']].plot(ax=axs[1])
        axs[1].set_title("Rolling Coefficients")
        plt.tight_layout()
        plt.show()
