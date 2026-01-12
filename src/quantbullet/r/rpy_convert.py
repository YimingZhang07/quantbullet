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
        BoolVector, IntVector, FloatVector, StrVector, ListVector
    )
    if isinstance(obj, NULLType):
        return None

    # Atomic vectors of length 1 -> scalar
    if isinstance(obj, (BoolVector, IntVector, FloatVector, StrVector)):
        if len(obj) == 1:
            return obj[0]
        else:
            return list(obj)

    # ListVector -> dict (recursive)
    if isinstance(obj, ListVector):
        raise NotImplementedError("ListVector to dict conversion not implemented yet")
    
    raise NotImplementedError("Conversion for this R object type not implemented yet")


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