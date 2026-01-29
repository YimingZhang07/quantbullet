import math
from typing import Any

import numpy as np


def to_json_value(value: Any) -> Any:
    """Convert values to strict JSON-compatible types.

    Supported:
    - None
    - str, bool, int, float (finite)
    - numpy scalars (bool_, integer, floating)
    - numpy arrays (recursively converted)
    - dict with string keys (recursively converted)
    - list/tuple (recursively converted)
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Float values must be finite for JSON serialization.")
        return value
    if isinstance(value, str):
        return value

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        fv = float(value)
        if not math.isfinite(fv):
            raise ValueError("Float values must be finite for JSON serialization.")
        return fv

    if isinstance(value, np.ndarray):
        return [to_json_value(v) for v in value.tolist()]

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"JSON object keys must be strings, got {type(k).__name__}.")
            out[k] = to_json_value(v)
        return out

    if isinstance(value, (list, tuple)):
        return [to_json_value(v) for v in value]

    raise TypeError(f"Unsupported type for JSON serialization: {type(value).__name__}")
