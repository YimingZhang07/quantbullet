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

def find_duplicate_columns(df: pd.DataFrame) -> list:
    """Return a list of duplicated column names in a DataFrame."""
    return df.columns[df.columns.duplicated()].tolist()

def drop_duplicate_columns(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """
    Drop duplicated columns in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    keep : {"first", "last"}, default "first"
        Which duplicate to keep:
        - "first": keep the first occurrence, drop others
        - "last": keep the last occurrence, drop others

    Returns:
    --------
    pd.DataFrame
        DataFrame with duplicate columns dropped.
    """
    return df.loc[:, ~df.columns.duplicated(keep=keep)]

def drop_selected_duplicate_columns(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """
    Drop duplicate columns by position, based on a provided list of column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (may contain duplicate column names).
    cols_to_drop : list
        List of column names to drop *only the duplicate occurrences* for.
        Keeps the first occurrence, drops later ones.

    Returns
    -------
    pd.DataFrame
        DataFrame with selected duplicates dropped.
    """
    keep_mask = []
    seen = {}

    for col in df.columns:
        if col in cols_to_drop:
            seen[col] = seen.get(col, 0) + 1
            # Keep only the first occurrence
            keep_mask.append(seen[col] == 1)
        else:
            keep_mask.append(True)

    return df.loc[:, keep_mask].copy()
