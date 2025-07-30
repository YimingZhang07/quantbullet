import pandas as pd

def aggregate_trades_flex(
    df, 
    group_by,
    weight_col=None,
    default_method='first', 
    overrides=None
):
    """
    Aggregates a DataFrame to one row per security with flexible aggregation rules.

    Parameters:
    - df: input DataFrame
    - group_by: column or list of columns to group by
    - weight_col: column used for weighted average
    - default_method: default aggregation method ('first', 'last', 'mean', 'sum')
    - overrides: dict mapping column name -> aggregation method
    
    Returns:
    - Aggregated DataFrame
    """
    import numpy as np

    def wavg(x) : return np.average(x, weights=x[weight_col])
    def first(x): return x.iloc[0]
    def last(x) : return x.iloc[-1]

    agg_funcs = {
        'first' : first,
        'last'  : last,
        'mean'  : 'mean',
        'sum'   : 'sum',
        'wavg'  : wavg,
    }

    overrides = overrides or {}

    # Handle single or multiple group_by columns
    group_by_cols = [group_by] if isinstance(group_by, str) else group_by

    # Sort by group_by to make 'first'/'last' meaningful
    df_sorted = df.sort_values(by=group_by_cols)

    agg_dict = {}
    for col in df.columns:
        if col in group_by_cols:
            continue
        method = overrides.get(col, default_method)
        if method == 'wavg' and weight_col is None:
            raise ValueError("Weight column must be specified for weighted average.")
        agg_func = agg_funcs[method]
        agg_dict[col] = agg_func

    grouped = df_sorted.groupby(group_by_cols).agg(agg_dict)
    return grouped.reset_index()

def collapse_duplicates(df, keys, agg_overrides=None):
    """
    Collapse duplicate rows by grouping on `keys`.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    keys : list[str]
        Columns to group by.
    agg_overrides : dict, optional
        Dict of column -> aggregation function, e.g. {"principal_paid": "sum"}.
        Defaults to {} (only overrides some columns).

    Returns
    -------
    pd.DataFrame
    """
    if agg_overrides is None:
        agg_overrides = {}
    
    # default: "first" for all non-key columns
    agg_dict = {col: "first" for col in df.columns if col not in keys}
    
    # override with user-specified rules
    agg_dict.update(agg_overrides)
    
    return df.groupby(keys, as_index=False).agg(agg_dict)
