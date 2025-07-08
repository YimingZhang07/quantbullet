import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ============================
# Below transformer is not yet completed
# I am thinking the transform method should be capable of handling multiple features just like other sklearn transformers.
# ============================

# class TruncatedPowerTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, knots, include_bias=False):
#         self.knots = np.asarray(knots)
#         if not np.all(np.diff(self.knots) > 0):
#             raise ValueError("Knots must be strictly increasing.")
#         self.include_bias = include_bias

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         """
#         Creates basis: [1, x, max(0, x - t1), ..., max(0, x - tk)]
#         """
#         X = np.asarray(X).ravel()

#         if self.include_bias:
#             basis =  [ np.ones_like(X), X ]
#         else:
#             basis = [ X ]
#         for t in self.knots:
#             basis.append(np.maximum(0, X - t))
#         return np.vstack(basis).T  # shape (n_samples, n_features)

#     def predict(self, X, beta):
#         return self.transform(X) @ beta

class FlatRampTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, knots, include_bias=False):
        self.knots = np.asarray(knots)
        if not np.all(np.diff(self.knots) > 0):
            raise ValueError("Knots must be strictly increasing.")
        self.include_bias = include_bias

    def fit(self, X, y=None):
        # check the shape of X, should be either 1D array of shape (n_samples,) or 2D array of shape (n_samples, 1)
        X = np.asarray(X)
        if X.ndim == 1:
            self.n_features_in_ = 1
        elif X.ndim == 2 and X.shape[1] == 1:
            self.n_features_in_ = 1
        else:
            raise ValueError("Input X must be a 1D array or a 2D array with one column.")
    
    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        basis = []
        if self.include_bias:
            basis.append(np.ones(X.shape[0]))

        for a, b in zip(self.knots[:-1], self.knots[1:]):
            # Create a flat ramp basis function
            basis_func = np.clip(X, a, b)
            basis.append(basis_func)

        return np.column_stack(basis)
