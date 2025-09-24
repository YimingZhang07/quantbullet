import pandas as pd

def sort_multiindex_by_hierarchy(df, row_orders=None, col_orders=None):
    """Sort a DataFrame with MultiIndex by specified row and column orders.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex on rows and/or columns.
    row_orders : dict, optional
        Dictionary specifying the desired order for each row index level.
        Keys can be level names or level indices (0-based).
        Example: {0: ['A', 'B', 'C'], 'Category': ['X', 'Y', 'Z']}
    col_orders : dict, optional
        Dictionary specifying the desired order for each column index level.
        Keys can be level names or level indices (0-based).
        Example: {0: ['Q1', 'Q2', 'Q3', 'Q4'], 'Region': ['North', 'South']}

    Returns
    -------
    pd.DataFrame
        DataFrame sorted according to the specified hierarchy.
    """
    df2 = df.copy()

    if row_orders:
        arrays = []
        for i in range(df2.index.nlevels):
            vals = df2.index.get_level_values(i)
            # try to get order by name first, then by level index
            name = df2.index.names[i]
            order = (
                row_orders.get(name)
                if name in row_orders
                else row_orders.get(i)
            )
            if order is not None:
                arrays.append(pd.Categorical(vals, categories=order, ordered=True))
            else:
                arrays.append(vals)
        df2.index = pd.MultiIndex.from_arrays(arrays, names=df2.index.names)
        df2 = df2.sort_index()

    if col_orders:
        arrays = []
        for i in range(df2.columns.nlevels):
            vals = df2.columns.get_level_values(i)
            name = df2.columns.names[i]
            order = (
                col_orders.get(name)
                if name in col_orders
                else col_orders.get(i)
            )
            if order is not None:
                arrays.append(pd.Categorical(vals, categories=order, ordered=True))
            else:
                arrays.append(vals)
        df2.columns = pd.MultiIndex.from_arrays(arrays, names=df2.columns.names)
        df2 = df2.sort_index(axis=1)

    return df2