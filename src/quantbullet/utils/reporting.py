import pandas as pd
from enum import Enum
from typing import List

def df_to_html_table(df: pd.DataFrame, table_title: str = '') -> str:
    """
    Function to convert a pandas DataFrame into an HTML table.

    Parameters
    ----------
    df : pd.DataFrame
    """

    # Turn the columns that are floats into strings with 2 decimal places
    for col in df.select_dtypes(include='float').columns:
        df.loc[:, col] = df[col].apply(lambda x: f'{x:.2f}')

    return f"""
    <div class="table-title">{table_title}</div>
    <table>
        <thead>
            <tr>{''.join(f'<th>{col}</th>' for col in df.columns)}</tr>
        </thead>
        <tbody>
            {''.join(f'<tr>{"".join(f"<td>{value}</td>" for value in row)}</tr>' for row in df.values)}
        </tbody>
    </table>
    """

# Function to generate the complete HTML page
def generate_html_page(body_content: str, table_col_width = "20%") -> str:

    return f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            <!-- h2 {{ color: #4CAF50; text-align: center; }} -->
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th {{ background-color: #005CAF; color: white; font-weight: bold; padding: 2px; text-align: right; width: {table_col_width}; }}
            td {{ padding: 2px; border-bottom: 1px solid #ddd; text-align: right; width: {table_col_width}; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .table-title {{ font-weight: normal; text-align: left; }}
        </style>
    </head>
    <body>
        {body_content}
    </body>
    </html>
    """

class DataType(Enum):
    DATE        = "date"
    STRING      = "string"
    FLOAT       = "float"
    INT         = "int"
    BOOL        = "bool"
    DATE_TIME   = "datetime"

def consolidate_data_types(df: pd.DataFrame, type_map: dict) -> pd.DataFrame:
    """
    Consolidates the data types in the DataFrame based on the provided type map.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be consolidated.
    type_map : dict
        A dictionary mapping column names to their respective data types.

    Returns
    -------
    pd.DataFrame
        The DataFrame with consolidated data types.
    """
    for column, dtype in type_map.items():
        if column in df.columns:
            if dtype == DataType.DATE:
                df[column] = pd.to_datetime(df[column], errors='coerce').dt.date
            elif dtype == DataType.STRING:
                df[column] = df[column].astype(str)
            elif dtype == DataType.FLOAT:
                df[column] = pd.to_numeric(df[column], errors='coerce').astype(float) # consider to use 'float64'
            elif dtype == DataType.INT:
                df[column] = pd.to_numeric(df[column], errors='coerce').astype(int) # consider to use 'Int64'
            elif dtype == DataType.BOOL:
                df[column] = df[column].astype(bool)
            elif dtype == DataType.DATE_TIME:
                df[column] = pd.to_datetime(df[column], errors='coerce')
    
    return df


def combine_latest_records(dataframes: List[pd.DataFrame], unique_keys: List[str]) -> pd.DataFrame:
    """
    Combines a sequence of DataFrames, keeping only the latest records based on unique keys.
    
    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        A list of DataFrames to be combined.
    unique_keys : List[str]
        A list of columns that uniquely identify each record.

    Returns
    -------
    pd.DataFrame
        The combined DataFrame containing the latest records.
    """
    if not dataframes:
        raise ValueError("The dataframes list cannot be empty.")
    if not unique_keys:
        raise ValueError("The unique_keys list cannot be empty.")
    
    # Initialize an empty master dataframe with columns from the first dataframe
    master_df = pd.DataFrame(columns=dataframes[0].columns)
    
    # Iterate over the list of dataframes
    for df in dataframes:
        if not all(key in df.columns for key in unique_keys):
            raise ValueError(f"All unique_keys must be present in each DataFrame. Missing in {df.columns}")
        
        # Combine and drop duplicates based on unique keys
        master_df = pd.concat([master_df, df]).drop_duplicates(subset=unique_keys, keep='last')
    
    # Reset the index before returning
    master_df = master_df.reset_index(drop=True)
    return master_df