import numpy as np
import re
import pandas as pd
from sklearn.utils.validation import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from quantbullet.dfutils import get_bins_and_labels

class FlatRampTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, knots, include_bias=False, dummy_segments=None, keep_ramp_for_dummies=False):
        self.knots = np.asarray(knots)
        if not np.all(np.diff(self.knots) > 0):
            raise ValueError("Knots must be strictly increasing.")
        self.include_bias = include_bias
        self.dummy_segments = dummy_segments or []
        self.keep_ramp_for_dummies = keep_ramp_for_dummies
        
    def _normalize_segment_indices(self, indices):
        """Convert dummy segment indices (with negatives) to 0..n_pieces-1."""
        if not indices:
            return set()
        normalized = set()
        for idx in indices:
            if idx < 0:
                idx = self.n_pieces_ + idx  # convert negative
            if idx < 0 or idx >= self.n_pieces_:
                raise ValueError(f"dummy_segments index {idx} out of range 0..{self.n_pieces_-1}")
            normalized.add(idx)
        return normalized

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
        self.n_pieces_ = len(self.knots) + 1
        self.dummy_segments_ = self._normalize_segment_indices(self.dummy_segments)
        
        fname = self.feature_names_in_[0]
        names = []

        if self.include_bias:
            names.append(f"{fname}_bias")

        # leftmost piece (index 0)
        a = self.numeric_to_string(self.knots[0])
        if 0 in self.dummy_segments_:
            if self.keep_ramp_for_dummies:
                names.append(f"{fname}_le_{a}")
            names.append(f"{fname}_le_{a}_d")
        else:
            names.append(f"{fname}_le_{a}")

        # middle pieces (1..n_knots-1)
        for seg_idx, (a, b) in enumerate(zip(self.knots[:-1], self.knots[1:]), start=1):
            sa, sb = self.numeric_to_string(a), self.numeric_to_string(b)
            if seg_idx in self.dummy_segments_:
                if self.keep_ramp_for_dummies:
                    names.append(f"{fname}_in_{sa}_{sb}")
                names.append(f"{fname}_in_{sa}_{sb}_d")
            else:
                names.append(f"{fname}_{sa}_{sb}")

        # rightmost piece (last index)
        last_idx = self.n_pieces_ - 1
        b = self.numeric_to_string(self.knots[-1])
        if last_idx in self.dummy_segments_:
            if self.keep_ramp_for_dummies:
                names.append(f"{fname}_gt_{b}")
            names.append(f"{fname}_gt_{b}_d")
        else:
            names.append(f"{fname}_gt_{b}")

        self.feature_names_out_ = np.array(names, dtype=object)
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

        if self.include_bias:
            basis.append(np.ones_like(x))

        # leftmost (upper cap)
        a = self.knots[0]
        if 0 in self.dummy_segments_:
            if self.keep_ramp_for_dummies:
                basis.append(np.clip(x, None, a))
            basis.append((x <= a).astype(float))
        else:
            basis.append(np.clip(x, None, a))

        # middle pieces
        for seg_idx, (a, b) in enumerate(zip(self.knots[:-1], self.knots[1:]), start=1):
            if seg_idx in self.dummy_segments_:
                if self.keep_ramp_for_dummies:
                    basis.append(np.clip(x, a, b))
                basis.append(((x >= a) & (x < b)).astype(float))
            else:
                basis.append(np.clip(x, a, b))

        # rightmost piece
        last_idx = self.n_pieces_ - 1
        b = self.knots[-1]
        if last_idx in self.dummy_segments_:
            if self.keep_ramp_for_dummies:
                basis.append(np.clip(x, b, None))
            basis.append((x >= b).astype(float))
        else:
            basis.append(np.clip(x, b, None))

        return np.column_stack(basis)

    def get_feature_names_out(self, input_features=None):
        return getattr(self, "feature_names_out_", None)

    def get_bins_and_labels(self):
        return get_bins_and_labels(list(self.knots))