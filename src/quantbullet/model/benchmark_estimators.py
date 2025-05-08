import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

class LastValueEstimator(BaseEstimator, RegressorMixin):
    """
    A simple regressor that always predicts the mean of the latest y values
    associated with the latest value in a reference column (e.g., a time column).

    Parameters
    ----------
    reference_column : str
        The name of the column in X to use for sorting (typically a timestamp).
    """
    def __init__(self, reference_column):
        self.reference_column = reference_column

    def fit(self, X, y):
        if self.reference_column not in X.columns:
            raise ValueError(f"Column '{self.reference_column}' not found in X.")

        # Sort by reference column
        sorted_indices = X[self.reference_column].argsort()
        X_sorted = X.iloc[sorted_indices]
        y_sorted = y.iloc[sorted_indices]

        # Compute mean of y at the latest timestamp
        last_ref_value = X_sorted[self.reference_column].iloc[-1]
        self.last_value_ = y_sorted[X_sorted[self.reference_column] == last_ref_value].mean()

        return self

    def predict(self, X):
        return np.full(len(X), self.last_value_)
