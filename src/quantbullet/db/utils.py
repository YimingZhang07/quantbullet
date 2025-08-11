from sqlalchemy import text
from sqlalchemy import inspect

def upsert_to_sqlserver(df, table_name, engine, pk_cols, schema="dbo"):
    """
    Upsert a DataFrame to a SQL Server table using MERGE.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to upsert.
    table_name : str
        The name of the target table in the database.
    engine : sqlalchemy.engine.Engine
        The SQLAlchemy engine connected to the database.
    pk_cols : list of str
        The primary key columns to match on for the upsert operation.
    schema : str, optional
        The schema of the target table (default is "dbo").

    Returns
    -------
    int
        The number of rows affected by the upsert operation.
    """
    insp = inspect(engine)
    cols_meta = insp.get_columns(table_name, schema=schema)
    cols_in_db = [col["name"] for col in cols_meta]

    if pk_cols is None:
        raise ValueError("Primary key columns (pk_cols) must be specified for upsert operation.")

    # Check for required columns missing; this only impacts if we are inserting new rows
    # required_cols = [c["name"] for c in cols_meta if not c["nullable"] and c.get("default") is None]
    # missing_required = [c for c in required_cols if c not in df.columns]
    # if missing_required:
    #     raise ValueError(f"Missing required non-nullable columns in df: {missing_required}")

    # Trim df to only known columns
    df = df[[c for c in df.columns if c in cols_in_db]]

    # Create a persistent staging table name
    staging_table = f"{schema}.{table_name}_staging"

    with engine.begin() as conn:
        # overwrite staging table
        df.to_sql(f"{table_name}_staging", conn, schema=schema, if_exists="replace", index=False)

        # Build MERGE
        update_cols = [c for c in df.columns if c not in pk_cols]
        set_clause = ", ".join([f"T.{c} = S.{c}" for c in update_cols]) if update_cols else ""

        insert_cols = ", ".join(df.columns)
        insert_vals = ", ".join([f"S.{c}" for c in df.columns])

        merge_sql = f"""
        MERGE {schema}.{table_name} AS T
        USING {staging_table} AS S
        ON {" AND ".join([f"T.{c} = S.{c}" for c in pk_cols])}
        WHEN MATCHED THEN
            {f"UPDATE SET {set_clause}" if set_clause else "/* nothing to update */"}
        WHEN NOT MATCHED BY TARGET THEN
            INSERT ({insert_cols}) VALUES ({insert_vals});
        DROP TABLE {staging_table};
        """

        conn.execute(text(merge_sql))

    return len(df)