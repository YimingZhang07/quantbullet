import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Union, Tuple, List, Optional, Any

from quantbullet.model.gam import (
    GAMTermData, 
    SplineTermData, 
    SplineByGroupTermData, 
    TensorTermData, 
    FactorTermData
)
from quantbullet.model.smooth_fit import make_monotone_predictor_pchip

class GAMReplayModel:
    """
    A lightweight GAM predictor that reconstructs the model logic from exported partial dependence data.
    
    This class does not depend on pygam for prediction. It uses interpolation (PCHIP for splines, 
    RegularGridInterpolator for tensors) and lookups (for factors) based on the data exported 
    by WrapperGAM.get_partial_dependence_data().
    """
    
    def __init__(self, term_data: Dict[Union[str, Tuple[str, str]], GAMTermData], intercept: float = 0.0):
        """
        Initialize the replay model.

        Parameters
        ----------
        term_data : dict
            The dictionary returned by WrapperGAM.get_partial_dependence_data().
        intercept : float
            The model intercept (bias).
        """
        self.term_data = term_data
        self.intercept = intercept
        self._predictors = {}
        self._build_predictors()

    def _build_predictors(self):
        """Build callable predictors/interpolators for each term."""
        for key, data in self.term_data.items():
            
            # ----------------------------
            # Spline Term
            # ----------------------------
            if isinstance(data, SplineTermData):
                # Use make_monotone_predictor_pchip for 1D interpolation
                # It handles extrapolation (defaults to flat)
                self._predictors[key] = make_monotone_predictor_pchip(
                    data.x, 
                    data.y, 
                    extrapolate='flat'
                )

            # ----------------------------
            # Spline By Group Term
            # ----------------------------
            elif isinstance(data, SplineByGroupTermData):
                # Create a dictionary of interpolators, one for each group
                group_preds = {}
                for group_label, curves in data.group_curves.items():
                    group_preds[group_label] = make_monotone_predictor_pchip(
                        curves['x'], 
                        curves['y'], 
                        extrapolate='flat'
                    )
                self._predictors[key] = group_preds

            # ----------------------------
            # Tensor Term
            # ----------------------------
            elif isinstance(data, TensorTermData):
                # RegularGridInterpolator requires points to be strictly increasing
                # We assume data.x and data.y are grids from meshgrid generation (sorted)
                # We want flat extrapolation, so we'll handle clipping in predict()
                # Store the bounds along with the interpolator
                self._predictors[key] = {
                    'interpolator': RegularGridInterpolator(
                        (data.x, data.y), 
                        data.z, 
                        method='pchip',
                        bounds_error=False, 
                        fill_value=None 
                    ),
                    'x_bounds': (data.x.min(), data.x.max()),
                    'y_bounds': (data.y.min(), data.y.max())
                }

            # ----------------------------
            # Factor Term
            # ----------------------------
            elif isinstance(data, FactorTermData):
                # Create a lookup dictionary
                # Map category label (str) to value
                self._predictors[key] = dict(zip(data.categories, data.values))
            
            else:
                raise ValueError(f"Unknown term type: {type(data)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict target values for the given data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features. Must contain all columns required by the terms.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        # Start with intercept
        n_samples = len(X)
        y_pred = np.full(n_samples, self.intercept, dtype=float)

        for key, data in self.term_data.items():
            if key not in self._predictors:
                continue

            predictor = self._predictors[key]

            # ----------------------------
            # Spline Term
            # ----------------------------
            if isinstance(data, SplineTermData):
                feat_name = data.feature
                if feat_name not in X.columns:
                    raise ValueError(f"Feature '{feat_name}' missing from input DataFrame.")
                
                x_vals = X[feat_name].values
                term_contrib = predictor(x_vals)
                y_pred += term_contrib

            # ----------------------------
            # Spline By Group Term
            # ----------------------------
            elif isinstance(data, SplineByGroupTermData):
                feat_name = data.feature
                by_name = data.by_feature
                if feat_name not in X.columns or by_name not in X.columns:
                    raise ValueError(f"Features '{feat_name}' or '{by_name}' missing from input DataFrame.")
                
                x_vals = X[feat_name].values
                by_vals = X[by_name].astype(str).values # Ensure string matching for keys
                
                term_contrib = np.zeros(n_samples, dtype=float)
                
                # Vectorized by group
                # Iterate over known groups in the predictor
                # For unknown groups in data, contribution remains 0 (standard GAM behavior for dummy interactions usually)
                for group_label, group_func in predictor.items():
                    mask = (by_vals == group_label)
                    if np.any(mask):
                        term_contrib[mask] = group_func(x_vals[mask])
                
                y_pred += term_contrib

            # ----------------------------
            # Tensor Term
            # ----------------------------
            elif isinstance(data, TensorTermData):
                feat_x = data.feature_x
                feat_y = data.feature_y
                if feat_x not in X.columns or feat_y not in X.columns:
                    raise ValueError(f"Features '{feat_x}' or '{feat_y}' missing from input DataFrame.")
                
                x_vals = X[feat_x].values
                y_vals = X[feat_y].values
                
                # Retrieve the predictor dict (interpolator + bounds)
                pred_obj = predictor
                interpolator = pred_obj['interpolator']
                x_min, x_max = pred_obj['x_bounds']
                y_min, y_max = pred_obj['y_bounds']
                
                # Manual clamping for flat extrapolation
                x_clamped = np.clip(x_vals, x_min, x_max)
                y_clamped = np.clip(y_vals, y_min, y_max)
                
                pts = np.column_stack([x_clamped, y_clamped])
                
                term_contrib = interpolator(pts)
                y_pred += term_contrib

            # ----------------------------
            # Factor Term
            # ----------------------------
            elif isinstance(data, FactorTermData):
                feat_name = data.feature
                if feat_name not in X.columns:
                    raise ValueError(f"Feature '{feat_name}' missing from input DataFrame.")
                
                vals = X[feat_name].astype(str).map(predictor)
                
                # Handle unknown categories -> 0.0 contribution
                vals = vals.fillna(0.0).values
                y_pred += vals

        return y_pred

