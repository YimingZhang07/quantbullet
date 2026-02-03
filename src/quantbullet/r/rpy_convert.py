import pandas as pd
from .r_session import get_r


def py_df_to_r(df: pd.DataFrame, r=None):
    if r is None:
        r = get_r()
    from rpy2.robjects.conversion import get_conversion
    with r.localconverter(r.ro.default_converter + r.pandas2ri.converter):
        return get_conversion().py2rpy(df)
    
def py_array_to_r(arr):
    r = get_r()
    with r.localconverter(r.ro.default_converter + r.numpy2ri.converter):
        return r.ro.conversion.py2rpy(arr)
    
def r_array_to_py(obj):
    r = get_r()
    from rpy2.robjects.conversion import get_conversion
    with r.localconverter(r.ro.default_converter + r.numpy2ri.converter):
        return get_conversion().rpy2py(obj)
    
def r_df_to_py(df_r):
    r = get_r()
    from rpy2.robjects.conversion import get_conversion
    with r.localconverter(r.ro.default_converter + r.pandas2ri.converter):
        return get_conversion().rpy2py(df_r)
    
def r_generic_types_to_py(obj):
    """
    Convert common rpy2 R objects into clean Python objects.
    """
    # NULL -> None
    from rpy2.rinterface_lib.sexp import NULLType
    from rpy2.robjects.vectors import (
        BoolVector, IntVector, FloatVector, StrVector, ListVector, DataFrame, Matrix
    )
    if isinstance(obj, NULLType):
        return None

    if isinstance(obj, DataFrame):
        return r_df_to_py(obj)

    if isinstance(obj, Matrix):
        return r_array_to_py(obj)

    # Atomic vectors of length 1 -> scalar
    if isinstance(obj, (BoolVector, IntVector, FloatVector, StrVector)):
        if len(obj) == 1:
            return obj[0]
        else:
            return list(obj)

    # ListVector -> dict (recursive)
    if isinstance(obj, ListVector):
        names = list(obj.names) if obj.names is not None else None
        if not names or all(n is None or n == "" for n in names):
            return [r_generic_types_to_py(v) for v in obj]
        out = {}
        for idx, v in enumerate(obj):
            key = names[idx] if names[idx] not in (None, "") else str(idx)
            out[key] = r_generic_types_to_py(v)
        return out
    
    # Fallback to numpy conversion where possible
    try:
        return r_array_to_py(obj)
    except Exception:
        raise NotImplementedError("Conversion for this R object type not implemented yet")


def py_obj_to_r(obj, r=None):
    """
    Convert common Python objects (including dict/list) to R objects.

    Useful for passing structured configs into R functions.
    """
    if r is None:
        r = get_r()

    from rpy2.rinterface import NULL
    from rpy2.robjects import ListVector, StrVector, IntVector, FloatVector, BoolVector

    if obj is None:
        return NULL

    if isinstance(obj, dict):
        return ListVector({k: py_obj_to_r(v, r) for k, v in obj.items()})

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return ListVector({})
        if all(isinstance(x, dict) for x in obj):
            return ListVector({str(i + 1): py_obj_to_r(x, r) for i, x in enumerate(obj)})
        if all(isinstance(x, str) for x in obj):
            return StrVector(obj)
        if all(isinstance(x, bool) for x in obj):
            return BoolVector(obj)
        if all(isinstance(x, int) and not isinstance(x, bool) for x in obj):
            return IntVector(obj)
        if all(isinstance(x, (int, float)) for x in obj):
            return FloatVector(obj)
        return ListVector({str(i + 1): py_obj_to_r(x, r) for i, x in enumerate(obj)})

    if isinstance(obj, str):
        return StrVector([obj])
    if isinstance(obj, bool):
        return BoolVector([obj])
    if isinstance(obj, int) and not isinstance(obj, bool):
        return IntVector([obj])
    if isinstance(obj, float):
        return FloatVector([obj])

    return obj


# def r_general_types_to_py(obj):
#     """
#     Convert common rpy2 R objects into clean Python objects.
#     """
#     # NULL -> None
#     if isinstance(obj, NULLType):
#         return None

#     # Atomic vectors of length 1 -> scalar
#     if isinstance(obj, (BoolVector, IntVector, FloatVector, StrVector)):
#         if len(obj) == 1:
#             return obj[0]
#         else:
#             return list(obj)

#     # ListVector -> dict (recursive)
#     if isinstance(obj, ListVector):
#         return {name: r_general_types_to_py(obj.rx2(name)) for name in obj.names}

#     # Fallback: use rpy2 converters (numpy / pandas)
#     try:
#         return _r_to_py(obj)
#     except Exception:
#         return obj