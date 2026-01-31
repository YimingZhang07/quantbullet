import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Union, Tuple, List, Optional, Any

from quantbullet.model.gam import (
    GAMTermData, 
    SplineTermData, 
    SplineByGroupTermData, 
    TensorTermData, 
    FactorTermData,
    load_partial_dependence_json,
    format_term_name,
    parse_term_name,
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

    @classmethod
    def from_partial_dependence_json(cls, path: str) -> "GAMReplayModel":
        """Load a GAMReplayModel from a standard JSON file."""
        term_data, intercept, _ = load_partial_dependence_json(path)
        return cls(term_data=term_data, intercept=intercept)

    @classmethod
    def from_fitted_model(
        cls, 
        model: Any, 
        curve_length: int = 200,
        width: float = 0.95,
    ) -> "GAMReplayModel":
        """
        Construct a GAMReplayModel from any fitted GAM that exports to JSON format.
        
        Communication happens through the unified JSON payload format - no model
        type inspection required. The model must have `export_partial_dependence_payload()`
        or `get_partial_dependence_data()` method.
        
        Parameters
        ----------
        model : WrapperGAM | MgcvBamWrapper | Any
            A fitted GAM model that can export partial dependence data.
        curve_length : int
            Number of points per smooth curve (default: 200).
        width : float
            Confidence interval width (default: 0.95 for 95% CI).
            
        Returns
        -------
        GAMReplayModel
            A replay model that can predict without the original fitting library.
            
        Examples
        --------
        >>> # Works with any model that exports to JSON format
        >>> replay = GAMReplayModel.from_fitted_model(fitted_gam)
        >>> predictions = replay.predict(new_df)
        """
        import tempfile
        import os
        
        # Use the unified JSON interface - write to temp file, read back
        # This ensures we go through the exact same serialization path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Model exports to JSON (unified interface)
            model.export_partial_dependence_json(
                path=temp_path,
                curve_length=curve_length,
                width=width,
            )
            # Load from JSON (unified interface)
            return cls.from_partial_dependence_json(temp_path)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

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
        res = self.decompose(X)
        return res['pred']

    def _format_term_name(self, data: GAMTermData, group_label: Optional[str] = None) -> str:
        """
        Format term names using standard convention: {type}__{feature}[__{by_feature}__{by_level}]
        
        Examples:
            s__age, s__age__level__B, f__level, te__x1__x2
        """
        if isinstance(data, SplineTermData):
            return format_term_name("s", data.feature)
        elif isinstance(data, SplineByGroupTermData):
            if group_label is not None:
                return format_term_name("s", data.feature, 
                                        by_feature=data.by_feature, by_level=group_label)
            return format_term_name("s", data.feature)
        elif isinstance(data, TensorTermData):
            return format_term_name("te", data.feature_x, feature2=data.feature_y)
        elif isinstance(data, FactorTermData):
            return format_term_name("f", data.feature)
        return "unknown"

    def decompose(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Decompose prediction into contributions from each term.
        
        Matches the interface of WrapperGAM.decompose().

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        dict
            Dictionary containing:
            - 'pred': np.ndarray of total predictions
            - 'intercept': float
            - 'term_contrib': pd.DataFrame where columns are term names and rows are samples
            - 'term_indices': list of keys (feature names/tuples) corresponding to columns
        """
        n_samples = len(X)
        term_contribs = {}
        ordered_keys = []
        
        for key, data in self.term_data.items():
            if key not in self._predictors:
                continue
                
            ordered_keys.append(key)
            predictor = self._predictors[key]
            
            # Use standardized naming convention
            col_name = self._format_term_name(data)

            # ----------------------------
            # Spline Term
            # ----------------------------
            if isinstance(data, SplineTermData):
                feat_name = data.feature
                if feat_name not in X.columns:
                    raise ValueError(f"Feature '{feat_name}' missing from input DataFrame.")
                
                x_vals = X[feat_name].values
                contrib = predictor(x_vals)
                term_contribs[col_name] = contrib

            # ----------------------------
            # Spline By Group Term
            # ----------------------------
            elif isinstance(data, SplineByGroupTermData):
                feat_name = data.feature
                by_name = data.by_feature
                if feat_name not in X.columns or by_name not in X.columns:
                    raise ValueError(f"Features '{feat_name}' or '{by_name}' missing from input DataFrame.")
                
                x_vals = X[feat_name].values
                by_vals = X[by_name].astype(str).values 
                
                for group_label, group_func in predictor.items():
                    # Use consistent naming: s(feature):by_feature=level
                    term_col_name = self._format_term_name(data, group_label=group_label)
                    
                    contrib = np.zeros(n_samples, dtype=float)
                    mask = (by_vals == group_label)
                    if np.any(mask):
                        contrib[mask] = group_func(x_vals[mask])
                        
                    term_contribs[term_col_name] = contrib

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
                
                pred_obj = predictor
                interpolator = pred_obj['interpolator']
                x_min, x_max = pred_obj['x_bounds']
                y_min, y_max = pred_obj['y_bounds']
                
                x_clamped = np.clip(x_vals, x_min, x_max)
                y_clamped = np.clip(y_vals, y_min, y_max)
                
                pts = np.column_stack([x_clamped, y_clamped])
                contrib = interpolator(pts)
                term_contribs[col_name] = contrib

            # ----------------------------
            # Factor Term
            # ----------------------------
            elif isinstance(data, FactorTermData):
                feat_name = data.feature
                if feat_name not in X.columns:
                    raise ValueError(f"Feature '{feat_name}' missing from input DataFrame.")
                
                vals = X[feat_name].astype(str).map(predictor)
                vals = vals.fillna(0.0).values
                term_contribs[col_name] = vals

        # Construct DataFrame
        term_contrib_df = pd.DataFrame(term_contribs, index=X.index)
        
        # Calculate total prediction
        # Sum of terms + intercept
        total_pred = term_contrib_df.sum(axis=1).values + self.intercept
        
        # Add metadata columns to dataframe (matching WrapperGAM behavior)
        term_contrib_df['intercept'] = self.intercept
        term_contrib_df['pred'] = total_pred
        
        return {
            "pred": total_pred,
            "intercept": self.intercept,
            "term_contrib": term_contrib_df,
            "term_indices": ordered_keys  # In Replay, these are the dictionary keys
        }

