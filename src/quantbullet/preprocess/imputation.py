import numpy as np
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
                # fallback: use last known coefficients if available
                if len(coefs) > 0 and not any(np.isnan(coefs[-1])):
                    last_intercept, last_slope = coefs[-1]
                    pred = last_intercept + last_slope * x.iloc[t]
                    y_pred.iloc[t] = pred
                    coefs.append((last_intercept, last_slope))
                else:
                    # still not enough to infer anything
                    coefs.append((np.nan, np.nan))
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

class SmoothSpreadImputer:
    """
    Impute missing values in a time series using a linear model with a smooth spread.
    
    This class solves the following optimization problem:
    min ||y - β * x - s||^2 + λ * ||Δs||^2
    where y is the target series, x is the anchor series, s is the spread, and λ is the regularization parameter.

    For spread, instead of pushing it down to zero, we assume it stays constant until the next observation.
    """
    def __init__(self, lam=10.0):
        """
        Parameters
        ----------
        lam : float
            Regularization strength for smoothing the spread between observed points.
            Higher = smoother, lower = more responsive.
        """
        self.lam = lam
        self.beta_ = None
        self.spread_ = None
        self.fitted_ = None
        self._raw_y = None
        self._raw_x = None

    def fit(self, y: pd.Series, x: pd.Series):
        """
        Fit the model to sparse y and complete x using smooth spread imputation.
        
        Parameters
        ----------
        y : pd.Series
            The dependent variable (target).
        x : pd.Series
            The independent variable (predictor).
        """
        assert isinstance(y, pd.Series) and isinstance(x, pd.Series)
        assert y.index.equals(x.index)

        self._raw_y = y.copy()
        self._raw_x = x.copy()

        idx = y.index
        T = len(y)
        mask_obs = y.notna().values
        obs_idx = np.where(mask_obs)[0]
        N_obs = len(obs_idx)

        yv = y.values
        xv = x.values

        # Initial guess
        init_beta = 0.0
        init_spread = np.zeros(N_obs)
        init_params = np.concatenate(([init_beta], init_spread))

        # Objective with lambda
        def objective(params):
            beta = params[0]
            s_obs = params[1:]

            # Reconstruct full spread with hard constraint
            s_full = np.zeros(T)
            s_full[obs_idx[0]] = s_obs[0]
            j = 1
            for t in range(obs_idx[0] + 1, T):
                if mask_obs[t]:
                    s_full[t] = s_obs[j]
                    j += 1
                else:
                    s_full[t] = s_full[t - 1]

            y_hat = beta * xv + s_full
            fit_loss = np.sum((yv[mask_obs] - y_hat[mask_obs]) ** 2)

            smooth_penalty = np.sum(np.diff(s_obs) ** 2)
            return fit_loss + self.lam * smooth_penalty

        # Solve
        res = minimize(objective, init_params, method='L-BFGS-B')

        # Reconstruct s_full from optimized s_obs
        # === NEW: handle extrapolation explicitly ===
        # Before first obs: use s_obs[0]
        beta = res.x[0]
        s_obs = res.x[1:]
        s_full = np.zeros(T)
        for t in range(0, obs_idx[0]):
            s_full[t] = s_obs[0]

        # Between obs: use s_obs + forward fill
        s_full[obs_idx[0]] = s_obs[0]
        j = 1
        for t in range(obs_idx[0] + 1, T):
            if mask_obs[t]:
                s_full[t] = s_obs[j]
                j += 1
            else:
                s_full[t] = s_full[t - 1]

        # After last obs: use last known s
        for t in range(obs_idx[-1] + 1, T):
            s_full[t] = s_obs[-1]

        self.beta_ = beta
        self.spread_ = pd.Series(s_full, index=idx)
        self.fitted_ = pd.Series(beta * xv + s_full, index=idx)

        return self

    def get_fitted(self) -> pd.Series:
        return self.fitted_

    def get_beta(self) -> float:
        return self.beta_

    def get_spread(self) -> pd.Series:
        return self.spread_

    def plot(self):
        assert self.fitted_ is not None, "Call .fit() first."
        plt.figure(figsize=(12, 6))
        self._raw_y.plot(label='Original y', lw=2)
        self._raw_x.plot(label='x (anchor)', alpha=0.5)
        self.fitted_.plot(label='Imputed ŷ', lw=2, color='orange')
        plt.title(f"Optimized Imputation with Spread Smoothing (β = {self.beta_:.3f}, λ = {self.lam})")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()