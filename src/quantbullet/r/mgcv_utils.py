"""
Utilities for converting GAM components between R and Python formats.

This module provides shared logic for:
- Converting extracted GAM components to GAMTermData format
- Normalizing R-exported JSON to match Python structures
- Converting R JSON exports to standard GAMReplayModel format
"""

from typing import Dict, Tuple, Union, Any
import pandas as pd
import numpy as np

from quantbullet.model.gam import (
    GAMTermData,
    SplineTermData,
    SplineByGroupTermData,
    FactorTermData,
    dump_partial_dependence_json,
)


def components_to_term_data(
    components: dict,
    z_value: float = 1.96,
) -> Dict[Union[str, Tuple[str, str]], GAMTermData]:
    """
    Convert extracted GAM components to GAMTermData format.
    
    Shared logic for both live R models (via extract_components) and 
    R-exported JSON files (via normalize_json_components).
    
    Parameters
    ----------
    components : dict
        Components dict with keys: smooths, parametric, xlevels.
        - smooths: dict of DataFrames with columns x, fit, se, var, by_var, by_level
        - parametric: DataFrame with columns term, Estimate, Std. Error
        - xlevels: dict mapping factor names to their levels
    z_value : float
        Z-value for confidence intervals (default: 1.96 for 95% CI).
        
    Returns
    -------
    dict
        Mapping feature names (or tuples) to GAMTermData objects.
    """
    term_data: Dict[Union[str, Tuple[str, str]], GAMTermData] = {}
    smooths = components.get('smooths', {})
    
    # Group smooths: separate simple smooths from "by" factor smooths
    by_smooths: Dict[Tuple[str, str], Dict[str, pd.DataFrame]] = {}
    simple_smooths: Dict[str, pd.DataFrame] = {}
    
    for label, curve_df in smooths.items():
        var = curve_df['var'].iloc[0]
        by_var = curve_df.get('by_var', pd.Series([None])).iloc[0]
        by_level = curve_df.get('by_level', pd.Series([None])).iloc[0]
        
        if by_var is not None and pd.notna(by_var):
            key = (var, by_var)
            if key not in by_smooths:
                by_smooths[key] = {}
            by_smooths[key][str(by_level)] = curve_df
        else:
            simple_smooths[var] = curve_df
    
    # Create SplineTermData for simple smooths
    for var, curve_df in simple_smooths.items():
        x = curve_df['x'].values
        y = curve_df['fit'].values
        se = curve_df['se'].values if 'se' in curve_df.columns else np.zeros_like(y)
        
        term_data[var] = SplineTermData(
            feature=var,
            x=x,
            y=y,
            conf_lower=y - z_value * se,
            conf_upper=y + z_value * se,
        )
    
    # Create SplineByGroupTermData for "by" factor smooths
    for (var, by_var), level_curves in by_smooths.items():
        group_curves = {}
        for level, curve_df in level_curves.items():
            x = curve_df['x'].values
            y = curve_df['fit'].values
            se = curve_df['se'].values if 'se' in curve_df.columns else np.zeros_like(y)
            group_curves[level] = {
                'x': x,
                'y': y,
                'conf_lower': y - z_value * se,
                'conf_upper': y + z_value * se,
            }
        
        term_data[(var, by_var)] = SplineByGroupTermData(
            feature=var,
            by_feature=by_var,
            group_curves=group_curves,
        )
    
    # Create FactorTermData for categorical factors
    param_df = components.get('parametric')
    xlevels = components.get('xlevels', {})
    
    if param_df is not None and xlevels:
        for factor_name, levels in xlevels.items():
            levels = list(levels)
            if not levels:
                continue
            
            values = np.zeros(len(levels))
            conf_lower = np.zeros(len(levels))
            conf_upper = np.zeros(len(levels))
            
            for i, level in enumerate(levels):
                term_name = f"{factor_name}{level}"
                match = param_df[param_df['term'] == term_name]
                
                if not match.empty:
                    est = match['Estimate'].iloc[0]
                    se = match['Std. Error'].iloc[0] if 'Std. Error' in match.columns else 0
                    values[i] = est
                    conf_lower[i] = est - z_value * se
                    conf_upper[i] = est + z_value * se
            
            term_data[factor_name] = FactorTermData(
                feature=factor_name,
                categories=levels,
                values=values,
                conf_lower=conf_lower,
                conf_upper=conf_upper,
            )
    
    return term_data


def normalize_json_components(json_data: dict) -> dict:
    """
    Normalize R-exported JSON components to match extract_components() output.
    
    Converts:
    - smooths: dict of lists -> dict of DataFrames
    - parametric: dict of lists -> DataFrame
    
    Parameters
    ----------
    json_data : dict
        Raw JSON data loaded from R's save_gam_components_json().
        
    Returns
    -------
    dict
        Normalized components with DataFrames instead of lists.
    """
    normalized = {
        'intercept': json_data.get('intercept', 0.0),
        'link': json_data.get('link', 'identity'),
        'xlevels': json_data.get('xlevels', {}),
    }
    
    # Convert smooths from dict-of-lists to dict-of-DataFrames
    smooths = {}
    for label, curve_data in json_data.get('smooths', {}).items():
        df_data = {}
        n_points = len(curve_data.get('x', []))
        for key in ['x', 'fit', 'se', 'var', 'by_var', 'by_level', 'term']:
            if key in curve_data:
                val = curve_data[key]
                # Expand scalars/single-item lists to match curve length
                if isinstance(val, list) and len(val) == n_points:
                    df_data[key] = val
                elif isinstance(val, list) and len(val) == 1:
                    df_data[key] = val * n_points
                else:
                    df_data[key] = [val] * n_points
        smooths[label] = pd.DataFrame(df_data)
    normalized['smooths'] = smooths
    
    # Convert parametric from dict-of-lists to DataFrame
    parametric = json_data.get('parametric', {})
    if parametric and 'term' in parametric:
        normalized['parametric'] = pd.DataFrame(parametric)
    else:
        normalized['parametric'] = None
    
    return normalized


def convert_r_components_to_standard_json(
    r_json_path: str,
    output_path: str,
    width: float = 0.95,
) -> Dict[str, Any]:
    """
    Convert R-exported raw components JSON to standard GAMReplayModel JSON format.
    
    This allows models fitted purely in R (via save_gam_components_json()) to be
    used with GAMReplayModel in Python.
    
    Parameters
    ----------
    r_json_path : str
        Path to JSON file exported from R's save_gam_components_json().
    output_path : str
        Path to write the standard format JSON.
    width : float
        Confidence interval width (default: 0.95 for 95% CI).
        
    Returns
    -------
    dict
        The standard format payload that was written.
        
    Examples
    --------
    In R:
        >>> fit <- bam(y ~ s(x1) + s(x2, by=level) + level, data=df)
        >>> save_gam_components_json(fit, "r_components.json")
    
    In Python:
        >>> from quantbullet.r.mgcv_utils import convert_r_components_to_standard_json
        >>> convert_r_components_to_standard_json("r_components.json", "model.json")
        >>> 
        >>> from quantbullet.model.gam_replay import GAMReplayModel
        >>> replay = GAMReplayModel.from_partial_dependence_json("model.json")
        >>> predictions = replay.predict(new_df)
    """
    import json
    
    z_value = 1.96 if width == 0.95 else abs(__import__('scipy.stats', fromlist=['norm']).norm.ppf((1 - width) / 2))
    
    # Load and normalize R components to match extract_components() format
    with open(r_json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    components = normalize_json_components(json_data)
    term_data = components_to_term_data(components, z_value)
    
    metadata = {
        'source': 'R_mgcv',
        'link': components.get('link', 'identity'),
    }
    
    return dump_partial_dependence_json(
        term_data=term_data,
        path=output_path,
        intercept=components.get('intercept', 0.0),
        metadata=metadata,
    )

