import pandas as pd
import datetime

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
    formats = ["%Y-%m-%d", "%Y%m%d"]

    if isinstance(input_date, datetime.date) and not isinstance(input_date, datetime.datetime):
        return input_date

    if isinstance(input_date, (datetime.datetime, pd.Timestamp)):
        return input_date.date()

    if isinstance(input_date, str):
        for fmt in formats:
            try:
                return datetime.datetime.strptime(input_date, fmt).date()
            except ValueError:
                continue

    raise ValueError(
        f"Unsupported date format or type: {input_date} (expected formats: {formats})"
    )

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