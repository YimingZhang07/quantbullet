import numpy as np
from scipy.interpolate import interp1d

class InterpolatedModel:
    default_model_name = "InterpolatedModel"
    def __init__(self, kind="linear", extrapolation="flat", model_name=None):
        """
        Parameters
        ----------
        x_grid : array-like
            Grid points for interpolation
        y_grid : array-like  
            Values at grid points
        kind : str, default "linear"
            Type of interpolation ('linear', 'cubic', etc.)
        extrapolation : str, default "linear"
            How to handle extrapolation:
            - "linear": Linear extrapolation following the slope
            - "flat": Flat extrapolation using nearest boundary value
        """
        self.kind = kind
        self.extrapolation = extrapolation
        self.model_name = model_name

    def fit(self, x, y):
        self.x_grid = np.asarray(x)
        self.y_grid = np.asarray(y)
        if self.extrapolation == "linear":
            self.interp = interp1d(self.x_grid, self.y_grid, kind=self.kind, fill_value="extrapolate")
        else:  # flat extrapolation
            self.interp = interp1d(self.x_grid, self.y_grid, kind=self.kind, 
                                 fill_value=(self.y_grid[0], self.y_grid[-1]), 
                                 bounds_error=False)
        return self

    @property
    def model_name(self):
        return self._model_name if self._model_name is not None else self.default_model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value
    
    @classmethod
    def from_model(cls, model, x_min, x_max, n_points=200, **kwargs):
        x_grid = np.linspace(x_min, x_max, n_points)
        y_grid = model.predict(x_grid)
        return cls(x_grid, y_grid, **kwargs)

    @classmethod
    def from_parametric_model(cls, model, n_points=200, **kwargs):
        x_grid = np.linspace(model.left_bound_, model.right_bound_, n_points)
        y_grid = model.predict(x_grid)
        return cls(x_grid, y_grid, **kwargs)

    def predict(self, x):
        return self.interp(x)

    def to_dict(self):
        return {
            "x_grid": self.x_grid.tolist(), 
            "y_grid": self.y_grid.tolist(),
            "extrapolation": self.extrapolation
        }

    @classmethod
    def from_dict(cls, d, **kwargs):
        extrapolation = d.get("extrapolation", "linear")
        return cls(np.array(d["x_grid"]), np.array(d["y_grid"]), 
                  extrapolation=extrapolation, **kwargs)
    
    def math_repr(self):
        return " - Interpolated Model - "