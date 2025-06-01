import numpy as np
import pandas as pd
from numpy.linalg import inv, cholesky
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import mean_squared_error

class FeatureScaledKNNRegressor(BaseEstimator, RegressorMixin):
    """a KNN regressor that allows for weighted features when calculating distances"""
    def __init__(self, n_neighbors=5, metrics='euclidean', weights='uniform', feature_weights=None):
        """Initialize the weightedDistanceKNRegressor.
        
        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighbors to use in the underlying KNN regressor.
        metrics : str, default='euclidean'
            The distance metric to use for the KNN regressor.
        weights : str, default='uniform'
            Weight function used in prediction. Possible values are 'uniform' or 'distance', or a callable.
            This determines how the neighbors are weighted in the prediction.
        feature_weights : array-like, shape (n_features,), default=None
            Weights for each feature. If None, all features are equally weighted.
            This scales the features before applying the KNN algorithm.
        """
        self.n_neighbors = n_neighbors
        self.metrics = metrics
        self.weights = weights
        self.feature_weights = feature_weights

    def _apply_feature_weights(self, X):
        if self.feature_weights is None:
            return X
        weights = np.asarray(self.feature_weights)
        if weights.shape[0] != X.shape[1]:
            raise ValueError("Feature weights must match number of features.")
        return X * weights
    
    def _apply_mahalanobis_transform(self, X):
        cov = np.cov(X.T)
        L = cholesky(inv(cov))
        return X @ L.T
    
    def fit(self, X, y):
        """Fit the model using the training data."""
        if isinstance(X, pd.DataFrame):
            self.X_features_in = X.columns
        elif isinstance(X, np.ndarray):
            self.X_features_in = [f"feature_{i}" for i in range(X.shape[1])]
            
        # After checking the input, X and y are converted to numpy arrays
        X, y = check_X_y(X, y)
        self.X_train_ = pd.DataFrame(X, columns=self.X_features_in)
        self.y_train_ = pd.Series(y)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.n_features_in_ = X_scaled.shape[1]
        X_weighted = self._apply_feature_weights(X_scaled)

        if self.metrics == 'mahalanobis':
            X_final = self._apply_mahalanobis_transform(X_weighted)
        elif self.metrics == 'euclidean':
            X_final = X_weighted
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'mahalanobis'.")

        self.knn_ = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            metric='euclidean',
            weights=self.weights
        )
        self.knn_.fit(X_final, y)

        return self
    
    def _transform_input(self, X):
        """
        Applies scaler, feature weighting, and Mahalanobis transform if specified.
        Returns the transformed X_final used for distance and prediction.
        """
        X = check_array(X)
        X_scaled = self.scaler_.transform(X)
        X_weighted = self._apply_feature_weights(X_scaled)

        if self.metrics == 'mahalanobis':
            return self._apply_mahalanobis_transform(X_weighted)
        return X_weighted

    def predict(self, X):
        X_final = self._transform_input(X)
        return self.knn_.predict(X_final)
    
    def predict_with_neighbors(self, X):
        X_final = self._transform_input(X)
        y_pred = self.knn_.predict(X_final)
        distances, indices = self.knn_.kneighbors(X_final)

        flat_rows = []

        for i, (yp, idxs, dists) in enumerate(zip(y_pred, indices, distances)):
            for rank, (idx, dist) in enumerate(zip(idxs, dists)):
                row = self.X_train_.iloc[idx].to_dict()
                row.update({
                    '_request_index': i,
                    '_prediction': yp,
                    '_neighbor_index': idx,
                    '_neighbor_rank': rank,
                    '_distance': dist,
                    '_neighbor_y': self.y_train_.iloc[idx]
                })
                flat_rows.append(row)

        return pd.DataFrame(flat_rows)
    
    def apply_scaler_to_data( self, X )-> pd.DataFrame:
        """Apply the scaler to the input data and return the scaled features.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data to be scaled.
            
        Returns
        -------
        pd.DataFrame
            The input data with additional columns for the scaled features.
        """
        # make additional columns for the scaled features
        X_scaled = self.scaler_.transform(X)
        scaled_features = pd.DataFrame(X_scaled, columns=[f"{col}_scaled" for col in self.X_features_in])
        
        # X may not be a DataFrame, need to convert it to one
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_features_in)
        
        return pd.concat([X.reset_index(drop=True), scaled_features.reset_index(drop=True)], axis=1)

    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))