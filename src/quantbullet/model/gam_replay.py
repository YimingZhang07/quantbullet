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
    load_partial_dependence_json
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
        term_data, intercept, _ = load_partial_dependence_json(path)
        return cls(term_data=term_data, intercept=intercept)

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

    def _format_term_name(self, data: GAMTermData) -> str:
        """Helper to format term names to match WrapperGAM convention."""
        # WrapperGAM convention:
        # spline: s(feat)
        # spline by group: s(feat)*by(group)
        # tensor: te(feat1,feat2)
        # factor: f(feat)
        
        if isinstance(data, SplineTermData):
            return f"s({data.feature})"
        elif isinstance(data, SplineByGroupTermData):
            return f"s({data.feature})*by({data.by_feature})"
        elif isinstance(data, TensorTermData):
            return f"te({data.feature_x},{data.feature_y})"
        elif isinstance(data, FactorTermData):
            return f"f({data.feature})"
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
                
                # WrapperGAM explodes this into multiple columns: s(feat)*by(level_A), s(feat)*by(level_B)...
                # We need to match that.
                
                for group_label, group_func in predictor.items():
                    # Construct column name for this specific group interaction
                    # WrapperGAM usually formats as s(feat)*by(dummy_col_name)
                    # We need to approximate the dummy column name.
                    # Usually it is just the level name if we are lucky, or has a prefix.
                    # But WrapperGAM _format_term_name uses design_columns_.
                    # Let's assume the standard naming: s(feat)*by(level)
                    # Since we don't have the exact original dummy column name, we construct best effort.
                    # Wait, WrapperGAM decompose uses: s({feat})*by({by_name})
                    # But for categorical interactions, pygam has multiple terms, each with a different 'by' column (dummy).
                    # The dummy column is named like "{by_feat}___{level}".
                    # So the term name in WrapperGAM will be "s(feat)*by({by_feat}___{level})"
                    
                    # We should match this convention: "{by_name}___{group_label}"
                    dummy_col_name = f"{by_name}___{group_label}"
                    term_col_name = f"s({feat_name})*by({dummy_col_name})"
                    
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

