import pandas as pd

def stack_dataframes(dfs):
    """
    Stacks multiple DataFrames into a single DataFrame.
    Ensures all DataFrames have the same columns before concatenation.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        List of DataFrames to stack.

    Returns
    -------
    pd.DataFrame
        A single DataFrame containing all rows from the input DataFrames.
        If the input list is empty, returns an empty DataFrame.
    """
    if not dfs:
        return pd.DataFrame()

    # Use the columns from the first DataFrame as reference
    reference_columns = set(dfs[0].columns)
    concat_dfs = []
    
    for i, df in enumerate(dfs, start=1):
        if set(df.columns) != reference_columns:
            raise ValueError(f"DataFrame at index {i} has different columns.")
        if not df.empty:
            concat_dfs.append(df)
    
    # Concatenate and reset index
    full_df = pd.concat(concat_dfs, ignore_index=True)
    return full_df