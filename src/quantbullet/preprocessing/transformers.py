import numpy as np
import re
import pandas as pd
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from quantbullet.dfutils import get_bins_and_labels

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
                f"{self.feature_names_in_[0]}: No values smaller than leftmost knot {self.knots[0]}. "
                "Leftmost basis would be constant."
            )
        if np.all(X <= self.knots[-1]):
            raise ValueError(
                f"{self.feature_names_in_[0]}: No values greater than rightmost knot {self.knots[-1]}. "
                "Rightmost basis would be constant."
            )

        self.n_features_in_ = 1
        return self
    
    @staticmethod
    def numeric_to_string(val, float_decimals=2, sci_thresh=(1e3, 1e-2)):
        """Convert a numeric value to a formatted string.

        Helps with cleaner feature names after splitting the basis.
        """
        if isinstance(val, float):
            # use scientific if beyond thresholds
            if abs(val) >= sci_thresh[0] or (0 < abs(val) < sci_thresh[1]):
                s = f"{val:.1e}"  # scientific, 1 decimal
            else:
                s = f"{val:.{float_decimals}f}".rstrip("0").rstrip(".")
        
        elif isinstance(val, int):
            if abs(val) >= sci_thresh[0]:
                s = f"{val:.1e}"  # scientific
            else:
                s = str(val)
        
        else:
            s = str(val)
        
        # clean up for feature names (replace spaces etc.)
        s = re.sub(r"[^a-zA-Z0-9\.\-e]+", "", s)
        return s
    
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
        a = self.numeric_to_string(a)
        names.append(f"{self.feature_names_in_[0]}_le_{a}")

        # middle segments
        for a, b in zip(self.knots[:-1], self.knots[1:]):
            basis.append(np.clip(x, a, b))
            a = self.numeric_to_string(a)
            b = self.numeric_to_string(b)
            names.append(f"{self.feature_names_in_[0]}_{a}_{b}")

        # rightmost (lower floor)
        b = self.knots[-1]
        basis.append(np.clip(x, b, None))  # floor at b
        b = self.numeric_to_string(b)
        names.append(f"{self.feature_names_in_[0]}_gt_{b}")

        self.feature_names_out_ = np.array(names, dtype=object)
        return np.column_stack(basis)

    def get_feature_names_out(self, input_features=None):
        return getattr(self, "feature_names_out_", None)

    def get_bins_and_labels(self):
        return get_bins_and_labels(list(self.knots))