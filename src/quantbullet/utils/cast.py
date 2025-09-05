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