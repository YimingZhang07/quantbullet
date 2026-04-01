import json
import numpy as np
from typing import Dict, Union, Tuple, Optional, Any
from quantbullet.utils.serialize import to_json_value

from .terms import (
    GAMTermData,
    SplineTermData,
    SplineByGroupTermData,
    TensorTermData,
    FactorTermData,
    _term_key_from_data,
)


def export_partial_dependence_payload(
    term_data: Dict[Union[str, Tuple[str, str]], GAMTermData],
    intercept: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    terms = [value.to_dict() for value in term_data.values()]
    payload = {
        "intercept": float(intercept),
        "terms": terms,
    }
    if metadata is not None:
        payload["metadata"] = to_json_value(metadata)
    return payload


def dump_partial_dependence_json(
    term_data: Dict[Union[str, Tuple[str, str]], GAMTermData],
    path: str,
    intercept: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
    indent: int = 2,
) -> Dict[str, Any]:
    payload = export_partial_dependence_payload(
        term_data=term_data,
        intercept=intercept,
        metadata=metadata,
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=True)
    return payload


def load_partial_dependence_json(
    path: str,
) -> Tuple[Dict[Union[str, Tuple[str, str]], GAMTermData], float, Optional[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    term_data = {}
    for entry in payload.get("terms", []):
        data = GAMTermData.from_dict(entry)
        key = _term_key_from_data(data)
        term_data[key] = data

    intercept = float(payload.get("intercept", 0.0))
    metadata = payload.get("metadata")
    return term_data, intercept, metadata


def center_partial_dependence(
    term_data: Dict[Union[str, Tuple[str, str]], GAMTermData],
    intercept: float,
) -> Tuple[Dict[Union[str, Tuple[str, str]], GAMTermData], float]:
    """
    Center partial dependence curves so each curve averages to approximately
    zero, absorbing the offsets into the intercept.

    Prediction invariant is preserved:
    ``y = intercept + sum(f_i) = (intercept + sum(offset_i)) + sum(f_i - offset_i)``

    For by-group spline terms, each group curve is centered independently and
    the per-group offsets are folded into an existing or new ``FactorTermData``
    keyed by the ``by_feature``.

    Parameters
    ----------
    term_data : dict
        Mapping returned by ``get_partial_dependence_data()``.
    intercept : float
        The original model intercept.

    Returns
    -------
    (centered_term_data, new_intercept) : tuple
        Deep-copied term data with centered curves, and the adjusted intercept.
    """
    new_data: Dict[Union[str, Tuple[str, str]], GAMTermData] = {}
    intercept_offset = 0.0
    # by_feature -> {group_label: accumulated_offset}
    by_feature_offsets: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Pass 1: center each term, collecting by-group offsets on the side
    # ------------------------------------------------------------------
    for key, td in term_data.items():

        if isinstance(td, SplineTermData):
            offset = float(np.mean(td.y))
            new_data[key] = SplineTermData(
                feature=td.feature,
                x=td.x.copy(),
                y=td.y - offset,
                conf_lower=td.conf_lower - offset if td.conf_lower is not None else None,
                conf_upper=td.conf_upper - offset if td.conf_upper is not None else None,
            )
            intercept_offset += offset

        elif isinstance(td, FactorTermData):
            offset = float(np.mean(td.values))
            new_data[key] = FactorTermData(
                feature=td.feature,
                categories=list(td.categories),
                values=td.values - offset,
                conf_lower=td.conf_lower - offset if td.conf_lower is not None else None,
                conf_upper=td.conf_upper - offset if td.conf_upper is not None else None,
            )
            intercept_offset += offset

        elif isinstance(td, TensorTermData):
            offset = float(np.mean(td.z))
            new_data[key] = TensorTermData(
                feature_x=td.feature_x,
                feature_y=td.feature_y,
                x=td.x.copy(),
                y=td.y.copy(),
                z=td.z - offset,
            )
            intercept_offset += offset

        elif isinstance(td, SplineByGroupTermData):
            new_curves: Dict[str, Dict[str, np.ndarray]] = {}
            by_feat = td.by_feature

            if by_feat not in by_feature_offsets:
                by_feature_offsets[by_feat] = {}

            for label, curves in td.group_curves.items():
                x = curves["x"]
                y = curves["y"]
                offset = float(np.mean(y))

                by_feature_offsets[by_feat].setdefault(label, 0.0)
                by_feature_offsets[by_feat][label] += offset

                new_curves[label] = {
                    "x": x.copy(),
                    "y": y - offset,
                    "conf_lower": (
                        curves["conf_lower"] - offset
                        if curves.get("conf_lower") is not None
                        else None
                    ),
                    "conf_upper": (
                        curves["conf_upper"] - offset
                        if curves.get("conf_upper") is not None
                        else None
                    ),
                }

            new_data[key] = SplineByGroupTermData(
                feature=td.feature,
                by_feature=td.by_feature,
                group_curves=new_curves,
            )

        else:
            new_data[key] = td

    # ------------------------------------------------------------------
    # Pass 2: fold by-group offsets into existing or new FactorTermData,
    #         then re-center so the factor still averages to zero.
    # ------------------------------------------------------------------
    for by_feat, group_offsets in by_feature_offsets.items():
        if by_feat in new_data and isinstance(new_data[by_feat], FactorTermData):
            existing = new_data[by_feat]
            feature_name = existing.feature
            categories = list(existing.categories)
            values = existing.values.copy()
            conf_lower = existing.conf_lower.copy() if existing.conf_lower is not None else None
            conf_upper = existing.conf_upper.copy() if existing.conf_upper is not None else None
            for i, cat in enumerate(categories):
                cat_str = str(cat)
                if cat_str in group_offsets:
                    values[i] += group_offsets[cat_str]
                    if conf_lower is not None:
                        conf_lower[i] += group_offsets[cat_str]
                    if conf_upper is not None:
                        conf_upper[i] += group_offsets[cat_str]
        else:
            feature_name = by_feat
            categories = sorted(group_offsets.keys())
            values = np.array([group_offsets[c] for c in categories])
            conf_lower = None
            conf_upper = None

        # Re-center: the absorbed offsets introduce a non-zero mean
        residual = float(np.mean(values))
        values = values - residual
        if conf_lower is not None:
            conf_lower = conf_lower - residual
        if conf_upper is not None:
            conf_upper = conf_upper - residual
        intercept_offset += residual

        new_data[by_feat] = FactorTermData(
            feature=feature_name,
            categories=categories,
            values=values,
            conf_lower=conf_lower,
            conf_upper=conf_upper,
        )

    new_intercept = intercept + intercept_offset
    return new_data, new_intercept
