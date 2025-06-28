import pandas as pd
from typing import List, Union, Tuple

def drop_columns_by_alias_group(
    df: pd.DataFrame,
    alias_groups: List[Union[Tuple[str, ...], List[str]]],
    keep: str = 'first'
) -> pd.DataFrame:
    """Drops columns from a DataFrame based on alias groups, keeping only the first or last column in each group.
    
    The motivation is that when joining multiple DataFrames, you may end up with columns that are essentially the same but have different names.
    """
    if keep not in {'first', 'last'}:
        raise ValueError("`keep` must be either 'first' or 'last'.")

    df = df.copy()

    for group in alias_groups:
        # Filter out columns that actually exist in the DataFrame
        existing = [col for col in group if col in df.columns]
        if len(existing) <= 1:
            continue  # nothing to drop

        # Decide which one to keep
        to_keep = existing[0] if keep == 'first' else existing[-1]
        to_drop = [col for col in existing if col != to_keep]

        df.drop(columns=to_drop, inplace=True)

    return df

def get_latest_n_per_group(df, group_col, date_col, n):
    """Returns the latest `n` records per group, sorted by a date column."""
    return (
        df.sort_values([group_col, date_col], ascending=[True, False])
          .groupby(group_col)
          .head(n)
          .reset_index(drop=True)
    )

def aggregate_trades_flex(
    df, 
    group_by='cusip', 
    weight_col='volume',
    default_method='first', 
    overrides=None
):
    """
    Aggregates a DataFrame to one row per security with flexible aggregation rules.

    Parameters:
    - df: input DataFrame
    - group_by: column to group by (e.g., 'cusip')
    - weight_col: column used for weighted average
    - default_method: default aggregation method ('first', 'last', 'mean', 'sum')
    - overrides: dict mapping column name -> aggregation method
                 (e.g., {'price': 'wavg', 'volume': 'sum', 'spread': 'mean'})
    
    Returns:
    - Aggregated DataFrame
    """
    import numpy as np
    import pandas as pd

    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df_sorted = df.sort_values(by='trade_date')

    # Define aggregation functions
    def wavg(x): return np.average(x, weights=x[weight_col])
    def first(x): return x.iloc[0]
    def last(x): return x.iloc[-1]

    # Default aggregator function map
    agg_funcs = {
        'first': first,
        'last': last,
        'mean': 'mean',
        'sum': 'sum',
        'wavg': wavg,
    }

    # Build aggregation dict
    overrides = overrides or {}
    agg_dict = {}
    for col in df.columns:
        if col == group_by:
            continue
        method = overrides.get(col, default_method)
        agg_func = agg_funcs[method]
        agg_dict[col] = agg_func

    grouped = df_sorted.groupby(group_by).agg(agg_dict)
    return grouped.reset_index()
