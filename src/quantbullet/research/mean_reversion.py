from ..model.linear_model import ar_ols
from ..utils.validation import are_columns_in_df

import numpy as np
import pandas as pd

class OrnsteinUhlenbeck:

    __slots__ = ['theta', 'mu', 'sigma', 'model']

    def __init__(self, theta: float = None, mu: float = None, sigma: float = None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return f"OrnsteinUhlenbeck(theta={self.theta}, mu={self.mu}, sigma={self.sigma})"

    @property
    def params_dict(self):
        return {
            'theta': self.theta,
            'mu': self.mu,
            'sigma': self.sigma
        }

    def fit(self, series):
        """
        Fit an Ornstein-Uhlenbeck model to a single time series using OLS regression.

        Parameters:
        -----------
            series (numpy.ndarray or pandas.Series): The time series data.

        Returns:
        --------
            None
        """
        self.model = ar_ols(series, lag_order=1)
        self.mu, self.theta, self.sigma = self._ou_params_from_ols_model(self.model)

    def _ou_params_from_ols_model(self, model):
        alpha_0 = model.params['const']
        alpha_1 = model.params['Lag_1']
        sigma = np.std(model.resid)
        return self._ou_params_from_ar1(alpha_0, alpha_1, sigma)

    def _ou_params_from_ar1(self, alpha_0, alpha_1, sigma):
        """
        Convert AR(1) model parameters to Ornstein-Uhlenbeck parameters.

        Parameters:
        -----------
            alpha_0 (float): The constant term in the AR(1) model.
            alpha_1 (float): The AR(1) coefficient.
            sigma (float): The standard deviation of the residuals.

        Returns:
        --------
            tuple: The mean reversion level, the mean reversion speed, and the volatility.
        """
        theta = 1 - alpha_1
        mu = alpha_0 / theta
        return mu, theta, sigma
    
    def generate_bands(self, df, num_std=2):
        """
        Generate upper and lower bands around the mean reversion level.

        Parameters:
        -----------
            df (pandas.DataFrame): The data to generate bands for.
            num_std (int): The number of standard deviations to use for the bands.

        Returns:
        --------
            pandas.DataFrame: The input data with upper and lower bands added.
        """
        # check columns mu, theta, sigma is in df
        if not are_columns_in_df(df, ['mu', 'theta', 'sigma']):
            raise ValueError('Columns mu, theta, sigma must be in the dataframe')
        
        df = df.copy()
        df['upper_band'] = df['mu'] + num_std * df['sigma'] / np.sqrt(2 * df['theta'] - df['theta']**2)
        df['lower_band'] = df['mu'] - num_std * df['sigma'] / np.sqrt(2 * df['theta'] - df['theta']**2)
        return df

def generate_band_based_signal(series, upper_band, lower_band):
    signal = pd.Series(index=series.index)
    signal[series >= upper_band] = -1
    signal[series <= lower_band] = 1
    return signal

class BollingerBands:

    __slots__ = ['window', 'num_std']

    def __init__(self, window: int = None, num_std: int = None):
        self.window = window
        self.num_std = num_std

    def __repr__(self):
        return f"BollingerBands(window={self.window}, num_std={self.num_std})"

    @property
    def params_dict(self):
        return {
            'window': self.window,
            'num_std': self.num_std
        }

    def generate_bands(self, series):
        rolling_mean = series.rolling(window=self.window).mean()
        rolling_std = series.rolling(window=self.window).std()
        upper_band = rolling_mean + (rolling_std * self.num_std)
        lower_band = rolling_mean - (rolling_std * self.num_std)
        df = pd.DataFrame({
            'price': series,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'deviation': (series - rolling_mean) / rolling_std,
            'signal': generate_band_based_signal(series, upper_band, lower_band)
        },
        index=series.index)
        return df
    