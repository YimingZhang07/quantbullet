import pandas as pd

def refresh_categories( df: pd.DataFrame ):
    """ for each categorical column, refresh the categories

    This is particularly useful for a polars dataframe that has been converted to pandas,
    as polars seems to combine all categories across all columns into one global category list.
    """
    for col in df.select_dtypes( include="category" ).columns:
        df[col] = df[col].astype( "str" ).astype( "category" )
    return df