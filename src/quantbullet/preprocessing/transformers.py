import numpy as np
import pandas as pd
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from quantbullet.dfutils import get_bins_and_labels

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
        # accept pandas Series or 1-column DataFrame
        if isinstance(X, pd.Series):
            self.feature_names_in_ = [X.name or "x0"]
            X = X.to_numpy()[:, None]
        elif isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("X must have exactly one feature (1 column).")
            self.feature_names_in_ = list(X.columns)
            X = X.to_numpy()
        else:
            X = check_array(X, ensure_2d=True, dtype=float)
            if X.shape[1] != 1:
                raise ValueError("X must have exactly one feature.")
            self.feature_names_in_ = ["x0"]

        if np.all(X >= self.knots[0]):
            raise ValueError(
                f"No values smaller than leftmost knot {self.knots[0]}. "
                "Leftmost basis would be constant."
            )
        if np.all(X <= self.knots[-1]):
            raise ValueError(
                f"No values greater than rightmost knot {self.knots[-1]}. "
                "Rightmost basis would be constant."
            )

        self.n_features_in_ = 1
        return self
    
    def transform(self, X):
        # keep same logic as fit
        if isinstance(X, pd.Series):
            X = X.to_numpy()[:, None]
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        else:
            X = check_array(X, ensure_2d=True, dtype=float)

        x = X.ravel()

        basis = []
        names = []

        if self.include_bias:
            basis.append(np.ones_like(x))
            names.append(f"{self.feature_names_in_[0]}_bias")

        # leftmost (upper cap)
        a = self.knots[0]
        basis.append(np.clip(x, None, a))  # cap above a
        names.append(f"{self.feature_names_in_[0]}_le_{a}")

        # middle segments
        for a, b in zip(self.knots[:-1], self.knots[1:]):
            basis.append(np.clip(x, a, b))
            names.append(f"{self.feature_names_in_[0]}_{a}_{b}")

        # rightmost (lower floor)
        b = self.knots[-1]
        basis.append(np.clip(x, b, None))  # floor at b
        names.append(f"{self.feature_names_in_[0]}_gt_{b}")

        self.feature_names_out_ = np.array(names, dtype=object)
        return np.column_stack(basis)

    def get_feature_names_out(self, input_features=None):
        return getattr(self, "feature_names_out_", None)

    def get_bins_and_labels(self):
        return get_bins_and_labels(list(self.knots))