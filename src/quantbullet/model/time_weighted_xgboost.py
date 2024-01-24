from functools import partial
import numpy as np
from xgboost import XGBRegressor
import warnings

# from sklearn.base import BaseEstimator

class TimeWeightedXGBRegressor(XGBRegressor):
    """XGBoost regressor with higher sample weights for recent data."""
    def __init__(self, reference_column:str='date', decay_rate=0.1, offset_days=30, **kwargs):
        if reference_column is None:
            raise TypeError("reference_column must be provided.")
        if decay_rate is None or offset_days is None:
            raise TypeError("decay_rate and offset_days must be provided.")
        self.reference_column = reference_column
        self.decay_rate = decay_rate
        self.offset_days = offset_days
        super().__init__(**kwargs)

    @staticmethod
    def logistic_weights(dates, decay_rate, offset_days):
        """Calculate weights using a logistic function.
        
        Parameters
        ----------
        dates : array-like
        decay_rate : float
            Decay rate for the logistic function.
        offset_days : int
            Offset in days for the logistic function.

        Returns
        -------
        weights : numpy.ndarray
            Array of weights.
        """
        # Calculate the number of days from the most recent date
        days = (dates.max() - dates).dt.days

        # Calculate weights using a logistic function
        weights = 1 / (1 + np.exp(decay_rate * (days - offset_days)))

        return weights

    def fit(self, X, y, **kwargs):
        # HACK: Suppress warnings from XGBoost during fitting
        # this is because addtional parameters are not used by parent class and will be warned
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.xgb_regressor = XGBRegressor(**super().get_params(deep=True))
            self.weight_func = partial(TimeWeightedXGBRegressor.logistic_weights,
                                decay_rate=self.decay_rate,
                                offset_days=self.offset_days)
            # Extract date column
            date = X.loc[:, self.reference_column ]
            X = X.drop(columns=[ self.reference_column ])
            weights = self.weight_func(date)
            # Call parent class's fit method with weights
            return self.xgb_regressor.fit(X, y, sample_weight=weights, **kwargs)
    
    def predict(self, X, *args, **kwargs):
        # Drop date column
        # NOTE the original X is not modified
        # Local X is a copy of the pointer pointing to the dataframe. 
        # We just assign a new value, making this pointer point to a new dataframe.
        X = X.drop(columns=[ self.reference_column ])

        # Call parent class's predict method
        return self.xgb_regressor.predict(X, *args, **kwargs)
