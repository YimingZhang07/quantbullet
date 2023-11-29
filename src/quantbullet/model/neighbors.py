import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

class weightedDistanceKNRegressor(BaseEstimator):
    def __init__(self, n_neighbors=20, weights=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
    
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        if self.weights is None:
            self.weights = np.ones(X_scaled.shape[1])
        X_weighted = X_scaled * np.array(self.weights)
    
    def predict(self, X):
        pass