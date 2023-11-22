from sklearn.base import BaseEstimator

class LastValueEstimator(BaseEstimator):
    """Last-value prediction for time series.
    
    This estimator always predicts the lastest value of y that it has seen.
    """
    def __init__(self, reference_column):
        self.reference_column = reference_column

    def fit(self, X, y=None):
        if self.reference_column not in X.columns:
            raise ValueError("The dataframe does not include the reference column.")
        sorted_indices = X[self.reference_column].argsort()
        X_sorted = X.iloc[sorted_indices]
        y_sorted = y.iloc[sorted_indices]
        
        # Store the mean of y_sorted values corresponding to the last date
        self.last_value_ = y_sorted[X_sorted[self.reference_column] ==\
                                     X_sorted[self.reference_column].iloc[-1]].mean()
        return self

    def predict(self, X):
        # Return an array with the same length as X, filled with the last value
        return [self.last_value_] * len(X)
