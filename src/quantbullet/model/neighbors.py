import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

class weightedDistanceKNRegressor(BaseEstimator):
    """a KNN regressor that allows for weighted features when calculating distances"""
    def __init__(self, n_neighbors=20, feature_weights=None):
        self.n_neighbors = n_neighbors
        self.feature_weights = feature_weights
    
    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.n_features_in_ = X_scaled.shape[1]
        self.feature_weights_ = self.feature_weights or np.ones(self.n_features_in_)
        if len(self.feature_weights_) != X_scaled.shape[1]:
            raise ValueError("weights must be of same length as number of features")
        X_weighted = X_scaled * np.array(self.feature_weights_)

        # use scaled and weighted features to fit knn
        self.knn_ = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.knn_.fit(X_weighted, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler_.transform(X)
        X_weighted = X_scaled * np.array(self.feature_weights_)
        return self.knn_.predict(X_weighted)
