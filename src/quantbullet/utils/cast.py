import math
import pandas as pd
import datetime
import numpy as np

def to_date(input_date):
    """
    Convert various date-like inputs into a datetime.date object.

    Parameters
    ----------
    input_date : str | datetime.date | datetime.datetime | pd.Timestamp
        Input date in different formats.
    formats : list[str], optional
        List of string formats to try, e.g. ["%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"].
        Defaults to ["%Y-%m-%d", "%Y%m%d"].

    Returns
    -------
    datetime.date

    Raises
    ------
    ValueError
        If the input cannot be parsed into a date.
    """

    if isinstance(input_date, np.datetime64):
        input_date = pd.Timestamp(input_date)

    if isinstance(input_date, datetime.date) and not isinstance(input_date, datetime.datetime):
        return input_date

    if isinstance(input_date, (datetime.datetime, pd.Timestamp)):
        return input_date.date()

    formats = ["%Y-%m-%d", "%Y%m%d"]

    if isinstance(input_date, str):
        for fmt in formats:
            try:
                return datetime.datetime.strptime(input_date, fmt).date()
            except ValueError:
                continue

    raise ValueError(
        f"Unsupported date format or type: {input_date} (type: {type(input_date).__name__})"
    )

def to_date_str(input_date, format="%Y%m%d"):
    """
    Convert various date-like inputs into a string representation.

    Parameters
    ----------
    input_date : str | datetime.date | datetime.datetime | pd.Timestamp
        Input date in different formats.
    format : str, optional
        The format string to use for the output date. Defaults to "%Y-%m-%d".

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If the input cannot be parsed into a date.
    """
    date_obj = to_date(input_date)
    return date_obj.strftime(format)

def df_columns_to_tuples(*args: pd.Series) -> list[tuple]:
    """
    Convert given DataFrame columns into a list of tuples.
    - If a Series is datetime-like, convert it to datetime.date
    - Otherwise leave unchanged
    """
    processed = []
    for s in args:
        if pd.api.types.is_datetime64_any_dtype(s):
            processed.append(s.dt.date.to_numpy())
        else:
            processed.append(s.to_numpy())
    
    # stack columns row-wise into tuples
    return list(map(tuple, np.column_stack(processed)))

def df_columns_to_dict(key_col: pd.Series, value_col: pd.Series) -> dict:
    """Build a dictionary from two DataFrame columns."""
    df = pd.DataFrame({"key": key_col, "val": value_col})

    # For each key, check how many distinct values there are
    conflicts = df.groupby("key")["val"].nunique()
    bad_keys = conflicts[conflicts > 1].index.tolist()
    if bad_keys:
        raise ValueError(f"Conflicting values found for keys: {bad_keys}")

    # Safe: reduce to unique pairs
    df_unique = df.drop_duplicates()
    return dict(zip(df_unique["key"], df_unique["val"]))

def _scalar_jsonable(v):
    # pandas missing
    if v is None:
        return None

    # numpy scalars
    if isinstance(v, (np.floating,)):
        fv = float(v)
        if math.isfinite(fv):
            return fv
        return None  # or str(fv)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)

    # python float inf/nan
    if isinstance(v, float):
        if math.isfinite(v):
            return v
        return None

    # pandas timestamp / datetime
    if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)):
        # isoformat keeps timezone if present
        return v.isoformat()

    # pandas timedelta
    if isinstance(v, (pd.Timedelta,)):
        return v.isoformat()  # e.g. 'P0DT00H00M01S'

    # pandas NA checks (avoid containers)
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    # plain JSON types
    if isinstance(v, (str, int, bool)):
        return v

    return str(v)

def to_jsonable(x):
    # DataFrame → JSON payload
    if isinstance(x, pd.DataFrame):
        records = x.to_dict(orient="records")
        return {
            "type": "dataframe",
            "data": [{k: to_jsonable(v) for k, v in row.items()} for row in records],
        }

    # Series
    if isinstance(x, pd.Series):
        return {k: to_jsonable(v) for k, v in x.to_dict().items()}

    # dict / list / tuple / set
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]

    # ndarray
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]

    return _scalar_jsonable(x)

import pandas as pd

def from_jsonable(x):
    """
    Lossy decoder for `to_jsonable`.

    - Rebuilds DataFrame payloads: {"type":"dataframe","data":[...]} -> pd.DataFrame
    - Recursively decodes dict/list
    - Leaves scalars as-is (strings stay strings; consumers decide)
    """

    # DataFrame payload
    if isinstance(x, dict) and x.get("type") == "dataframe" and "data" in x:
        data = x["data"]
        # data should be a list[dict]; still decode recursively in case nested payloads exist
        rows = [from_jsonable(r) for r in data] if isinstance(data, list) else []
        return pd.DataFrame(rows)

    # dict
    if isinstance(x, dict):
        return {k: from_jsonable(v) for k, v in x.items()}

    # list
    if isinstance(x, list):
        return [from_jsonable(v) for v in x]

    # scalar (int/float/bool/None/str already JSON-native)
    return x
