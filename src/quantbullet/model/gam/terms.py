import numpy as np
from dataclasses import dataclass, fields
from typing import List, Dict, Union, Tuple, Optional, Any
from quantbullet.utils.serialize import to_json_value


def _maybe_array(value: Optional[Any]) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value)


def _term_key_from_data(data: "GAMTermData") -> Union[str, Tuple[str, str]]:
    if isinstance(data, SplineTermData):
        return data.feature
    if isinstance(data, FactorTermData):
        return data.feature
    if isinstance(data, SplineByGroupTermData):
        return (data.feature, data.by_feature)
    if isinstance(data, TensorTermData):
        return (data.feature_x, data.feature_y)
    raise ValueError(f"Unknown term data type: {type(data)}")


# =============================================================================
# Term Name Formatting/Parsing Utilities
# =============================================================================
# Format: {type}__{feature}[__{by_feature}__{by_level}]
# Examples:
#   s__age              -> spline on age
#   s__age__level__B    -> spline on age, by level=B
#   f__level            -> factor on level
#   te__x1__x2          -> tensor on x1, x2

def format_term_name(
    term_type: str,
    feature: str,
    by_feature: Optional[str] = None,
    by_level: Optional[str] = None,
    feature2: Optional[str] = None,
) -> str:
    """
    Format a term name using the standard convention.
    
    Parameters
    ----------
    term_type : str
        Term type: 's' (spline), 'f' (factor), 'te' (tensor)
    feature : str
        Main feature name
    by_feature : str, optional
        For spline-by-group terms, the grouping variable
    by_level : str, optional
        For spline-by-group terms, the specific level
    feature2 : str, optional
        For tensor terms, the second feature
        
    Returns
    -------
    str
        Formatted term name like 's__age' or 's__age__level__B'
    """
    if term_type == "te" and feature2:
        return f"te__{feature}__{feature2}"
    if by_feature and by_level:
        return f"{term_type}__{feature}__{by_feature}__{by_level}"
    return f"{term_type}__{feature}"


def parse_term_name(name: str) -> Dict[str, str]:
    """
    Parse a term name back to its components.
    
    Parameters
    ----------
    name : str
        Term name like 's__age' or 's__age__level__B'
        
    Returns
    -------
    dict
        Dictionary with keys: type, feature, and optionally by_feature, by_level, or feature2
        
    Examples
    --------
    >>> parse_term_name('s__age')
    {'type': 's', 'feature': 'age'}
    
    >>> parse_term_name('s__age__level__B')
    {'type': 's', 'feature': 'age', 'by_feature': 'level', 'by_level': 'B'}
    
    >>> parse_term_name('te__x1__x2')
    {'type': 'te', 'feature': 'x1', 'feature2': 'x2'}
    """
    parts = name.split("__")
    
    if len(parts) < 2:
        raise ValueError(f"Invalid term name format: {name}")
    
    term_type = parts[0]
    result = {"type": term_type, "feature": parts[1]}
    
    if term_type == "te" and len(parts) == 3:
        result["feature2"] = parts[2]
    elif len(parts) == 4:
        result["by_feature"] = parts[2]
        result["by_level"] = parts[3]
    
    return result


@dataclass
class GAMTermData:
    """Base class for GAM term partial dependence data."""
    term_type: str = "base"

    def to_dict(self) -> Dict[str, Any]:
        raw = {f.name: getattr(self, f.name) for f in fields(self)}
        return {k: to_json_value(v) for k, v in raw.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GAMTermData":
        term_type = data.get("term_type")
        if term_type == "spline":
            return SplineTermData.from_dict(data)
        if term_type == "spline_by_category":
            return SplineByGroupTermData.from_dict(data)
        if term_type == "tensor":
            return TensorTermData.from_dict(data)
        if term_type == "factor":
            return FactorTermData.from_dict(data)
        raise ValueError(f"Unknown term_type: {term_type}")


@dataclass
class SplineTermData(GAMTermData):
    """Data for a simple spline term s(x)."""
    feature: str = ""
    x: np.ndarray = None
    y: np.ndarray = None
    conf_lower: np.ndarray = None
    conf_upper: np.ndarray = None
    term_type: str = "spline"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplineTermData":
        return cls(
            feature=data.get("feature", ""),
            x=_maybe_array(data.get("x")),
            y=_maybe_array(data.get("y")),
            conf_lower=_maybe_array(data.get("conf_lower")),
            conf_upper=_maybe_array(data.get("conf_upper")),
            term_type=data.get("term_type", "spline"),
        )


@dataclass
class SplineByGroupTermData(GAMTermData):
    """Data for a spline term interacted with a categorical s(x, by=cat)."""
    feature: str = ""
    by_feature: str = ""
    # map group_label -> {'x': np.ndarray, 'y': np.ndarray, 'conf_lower': np.ndarray, 'conf_upper': np.ndarray}
    group_curves: Dict[str, Dict[str, np.ndarray]] = None
    term_type: str = "spline_by_category"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SplineByGroupTermData":
        group_curves = data.get("group_curves")
        if group_curves is not None:
            group_curves = {
                label: {k: _maybe_array(v) for k, v in curves.items()}
                for label, curves in group_curves.items()
            }
        return cls(
            feature=data.get("feature", ""),
            by_feature=data.get("by_feature", ""),
            group_curves=group_curves,
            term_type=data.get("term_type", "spline_by_category"),
        )


@dataclass
class TensorTermData(GAMTermData):
    """Data for a tensor product term te(x, y)."""
    feature_x: str = ""
    feature_y: str = ""
    x: np.ndarray = None
    y: np.ndarray = None
    z: np.ndarray = None
    term_type: str = "tensor"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorTermData":
        return cls(
            feature_x=data.get("feature_x", ""),
            feature_y=data.get("feature_y", ""),
            x=_maybe_array(data.get("x")),
            y=_maybe_array(data.get("y")),
            z=_maybe_array(data.get("z")),
            term_type=data.get("term_type", "tensor"),
        )


@dataclass
class FactorTermData(GAMTermData):
    """Data for a categorical factor term f(cat)."""
    feature: str = ""
    categories: List[str] = None
    values: np.ndarray = None
    conf_lower: np.ndarray = None
    conf_upper: np.ndarray = None
    term_type: str = "factor"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactorTermData":
        categories = data.get("categories")
        if categories is not None:
            categories = list(categories)
        return cls(
            feature=data.get("feature", ""),
            categories=categories,
            values=_maybe_array(data.get("values")),
            conf_lower=_maybe_array(data.get("conf_lower")),
            conf_upper=_maybe_array(data.get("conf_upper")),
            term_type=data.get("term_type", "factor"),
        )
