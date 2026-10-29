import numpy as np
import re
import pandas as pd
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from quantbullet.dfutils import get_bins_and_labels


def _single_column(X):
    """One feature as a finite float vector and its name: a Series, a one-column DataFrame or an (n, 1) array."""
    if isinstance(X, pd.Series):
        name, values = (X.name if X.name is not None else "x0"), X.to_numpy()[:, None]
    elif isinstance(X, pd.DataFrame):
        if X.shape[1] != 1:
            raise ValueError("X must have exactly one feature (1 column).")
        name, values = X.columns[0], X.to_numpy()
    else:
        name, values = "x0", X
    values = check_array(values, ensure_2d=True, dtype=float)
    if values.shape[1] != 1:
        raise ValueError("X must have exactly one feature.")
    return values[:, 0], str(name)


class FlatRampTransformer(BaseEstimator, TransformerMixin):
    """Flat ramp basis: ``min(x, k_1)``, ``clip(x, k_1, k_2)``, ..., ``max(x, k_K)``, after an optional column of ones.

    Parameters
    ----------
    knots : array-like
        The knots of the flat ramp basis, strictly increasing.
    include_bias : bool, default=False
        Whether to include a bias term.
    """
    def __init__(self, knots, include_bias=False):
        self.knots = np.asarray(knots)
        if not np.all(np.diff(self.knots) > 0):
            raise ValueError("Knots must be strictly increasing.")
        self.include_bias = include_bias

    def fit(self, X, y=None):
        x, fname = _single_column(X)
        if np.all(x >= self.knots[0]):
            raise ValueError(
                f"{fname}: No values smaller than leftmost knot {self.knots[0]}. "
                "Leftmost basis would be constant."
            )
        if np.all(x <= self.knots[-1]):
            raise ValueError(
                f"{fname}: No values greater than rightmost knot {self.knots[-1]}. "
                "Rightmost basis would be constant."
            )

        self.feature_names_in_ = [fname]
        self.n_features_in_ = 1

        names = [f"{fname}_bias"] if self.include_bias else []
        names.append(f"{fname}_le_{self.numeric_to_string(self.knots[0])}")
        for a, b in zip(self.knots[:-1], self.knots[1:]):
            names.append(f"{fname}_{self.numeric_to_string(a)}_{self.numeric_to_string(b)}")
        names.append(f"{fname}_gt_{self.numeric_to_string(self.knots[-1])}")
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
        check_is_fitted(self, "feature_names_out_")
        x, _ = _single_column(X)
        basis = [np.ones_like(x)] if self.include_bias else []
        basis.append(np.clip(x, None, self.knots[0]))
        for a, b in zip(self.knots[:-1], self.knots[1:]):
            basis.append(np.clip(x, a, b))
        basis.append(np.clip(x, self.knots[-1], None))
        return np.column_stack(basis)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def get_bins_and_labels(self):
        return get_bins_and_labels(list(self.knots))


class TentTransformer(BaseEstimator, TransformerMixin):
    """Tent basis (degree-1 B-spline): one column per knot.

    Column ``j`` is 1 at knot ``j`` and falls linearly to 0 at the neighbouring knots; beyond the end knots
    the end column stays at 1. Every row sums to 1, so a curve ``basis @ coef`` takes the value ``coef[j]``
    at knot ``j``, is linear between knots and flat outside them. The basis equals sklearn's
    ``SplineTransformer(degree=1, extrapolation="constant")`` with the same knots. Rows sum to exactly 1 in
    the output ``dtype`` (``transform(X, dtype=np.float32)`` too), so shifting all coefficients by a constant
    shifts the curve by exactly that constant.

    Parameters
    ----------
    knots : array-like
        At least two, strictly increasing; checked by ``fit``.
    """

    def __init__(self, knots):
        self.knots = knots

    def fit(self, X, y=None):
        """Check the knots and that every column has data; names columns ``{feature}_at_{knot}``."""
        self._fit(X)
        return self

    def fit_transform(self, X, y=None, dtype=np.float64):
        """``fit`` and ``transform`` with one basis computation."""
        return self._fit(X, dtype)

    def _fit(self, X, dtype=np.float64):
        x, name = _single_column(X)
        knots = np.asarray(self.knots, dtype=float)
        if knots.ndim != 1 or len(knots) < 2 or not np.all(np.isfinite(knots)):
            raise ValueError("knots must be a one-dimensional array of at least two finite values.")
        if not np.all(np.diff(knots) > 0):
            raise ValueError("knots must be strictly increasing.")
        basis = self._basis(x, knots, dtype)
        empty = ~np.any(basis > 0, axis=0)
        if empty.any():
            raise ValueError(f"{name}: no values between the neighbours of knots {knots[empty].tolist()}; "
                             "their coefficients would not be identified.")
        self.knots_ = knots
        self.feature_names_in_ = [name]
        self.n_features_in_ = 1
        self.feature_names_out_ = np.array(
            [f"{name}_at_{np.format_float_positional(knot, trim='-')}" for knot in knots], dtype=object)
        return basis

    @staticmethod
    def _basis(x, knots, dtype=np.float64):
        x = np.clip(x, knots[0], knots[-1])
        left = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(knots) - 2)
        t = (x - knots[left]) / (knots[left + 1] - knots[left])
        # The larger weight is in [0.5, 1], so 1 minus it is exact (Sterbenz) and each row sums to exactly 1.
        larger = np.maximum(t, 1 - t).astype(dtype)
        smaller = np.asarray(1, dtype=dtype) - larger
        right_larger = t >= .5
        basis = np.zeros((len(x), len(knots)), dtype=dtype)
        rows = np.arange(len(x))
        basis[rows, left] = np.where(right_larger, smaller, larger)
        basis[rows, left + 1] = np.where(right_larger, larger, smaller)
        return basis

    def transform(self, X, dtype=np.float64):
        check_is_fitted(self, "knots_")
        x, _ = _single_column(X)
        return self._basis(x, self.knots_, dtype)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_