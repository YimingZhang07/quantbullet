from scipy.interpolate import splrep, BSpline
import numpy as np
from .base import ParametricModel


class SplineModel(ParametricModel):
    """
    Spline-based smooth curve model that passes through (or near) given anchor points.

    Uses scipy.interpolate.UnivariateSpline.

    Parameters
    ----------
    smoothing : float, optional
        Smoothing factor s (default 0 means exact interpolation through all points).
    k : int, optional
        Degree of spline (default cubic, k=3).
    """

    default_model_name = "SplineModel"

    def __init__(self, smoothing=0.0, k=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoothing = smoothing
        self.k = k
        self._spline = None

    # --- fitting ---
    def fit(self, x, y, weights=None, left_bound=None, right_bound=None, **kwargs):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.left_bound_ = np.min(x) if left_bound is None else left_bound
        self.right_bound_ = np.max(x) if right_bound is None else right_bound

        # Directly compute the spline representation tuple (t, c, k)
        # s=0 means exact interpolation; s>0 gives smoothing
        t, c, k = splrep(x, y, w=weights, s=self.smoothing, k=self.k)

        # Build BSpline manually so that .t, .c, .k always exist
        self._spline = BSpline(t, c, k)

        # Store parameters for serialization
        self.params_dict = {"t": t.tolist(), "c": c.tolist(), "k": int(k)}
        return self


    # --- prediction ---
    def func_with_kwargs(self, x, **params_dict):
        if self._spline is None:
            # reconstruct from params_dict if loaded from saved state
            from scipy.interpolate import BSpline
            t = np.asarray(params_dict["t"])
            c = np.asarray(params_dict["c"])
            k = int(params_dict.get("k", 3))
            self._spline = BSpline(t, c, k)
        return self._spline(x)

    def func_with_args(self, x, *args):
        # For spline model, args are ignored; prediction uses stored spline
        return self.func_with_kwargs(x, **self.params_dict)

    def get_param_names(self):
        # Knots and coeffs are stored as lists; treat as one composite parameter
        return ["t", "c", "k"]

    def math_repr(self):
        return (
            "f(x) = Spline interpolation fitted through anchor points "
            f"(degree={self.k}, smoothing={self.smoothing})"
        )

    def to_dict(self):
        if hasattr(self, "_spline"):
            if hasattr(self._spline, "t"):
                t = self._spline.t
                c = self._spline.c
                k = self._spline.k
            else:
                # fallback if still InterpolatedUnivariateSpline
                t = self._spline.get_knots()
                c = getattr(self._spline, "get_coeffs", lambda: [])()
                k = getattr(self, "k", 3)
            params_dict = {"t": t.tolist(), "c": c.tolist(), "k": k}
        else:
            params_dict = getattr(self, "params_dict", {})

        return {
            "params_dict": params_dict,
            "allow_extrapolation": self.allow_extrapolation,
            "left_bound_": float(self.left_bound_),
            "right_bound_": float(self.right_bound_),
            "_model_name": getattr(self, "_model_name", "SplineModel"),
            "smoothing": getattr(self, "smoothing", 0.0),
            "k": getattr(self, "k", 3)
        }

    @classmethod
    def from_dict(cls, data_dict):
        instance = cls(
            smoothing=data_dict.get("smoothing", 0.0),
            k=data_dict.get("k", 3),
            params_dict=data_dict.get("params_dict"),
            allow_extrapolation=data_dict.get("allow_extrapolation", False),
            model_name=data_dict.get("_model_name", data_dict.get("model_name")),
        )
        if "left_bound_" in data_dict:
            instance.left_bound_ = data_dict["left_bound_"]
        if "right_bound_" in data_dict:
            instance.right_bound_ = data_dict["right_bound_"]
        return instance
