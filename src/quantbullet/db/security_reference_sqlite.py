import pandas as pd
from sqlalchemy import (
    create_engine, Column, String, MetaData, Table, select, Boolean, DateTime, func
)
from sqlalchemy.dialects.sqlite import insert
from pathlib import Path
from typing import Union, List

from quantbullet.core.enums import StrEnum

class MappingCacheTableEnum(StrEnum):
    CUSIP   = 'Cusip'
    ISIN    = 'ISIN'
    TICKER  = 'Ticker'
    LAST_UPDATED = 'LastUpdated'

class SimpleMappingCache:
    def __init__(self, cache_dir: str = "security_cache", engine=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        if engine is not None:
            self.engine = engine
        else:
            db_path = self.cache_dir / "mappings.db"
            self.engine = create_engine(f'sqlite:///{db_path}')

        self._init_database()
    
    def _init_database(self):
        """Create table using SQLAlchemy"""
        metadata = MetaData()
        self.mappings_table = Table(
            'mappings', metadata,
            Column('cusip', String, primary_key=True),
            Column('isin', String, nullable=True),
            Column('ticker', String, nullable=True),
            Column('last_updated', DateTime, nullable=False, server_default=func.current_timestamp(), onupdate=func.current_timestamp()),
        )

        self.invalid_table = Table(
            'invalid_identifiers', metadata,
            Column('identifier', String, primary_key=True),
            Column('first_seen', DateTime, nullable=False, server_default=func.current_timestamp()),
        )

        metadata.create_all(self.engine)

    def record_invalid_identifier(self, identifiers: Union[str, List[str]]):
        """Record invalid identifiers in the database."""
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        
        records = [{'identifier': ident} for ident in identifiers]
        stmt = insert(self.invalid_table).values(records)
        stmt = stmt.on_conflict_do_nothing(index_elements=['identifier'])
        with self.engine.begin() as conn:
            conn.execute(stmt)
    
    def add_mappings(self, df: pd.DataFrame):
        """Bulk upsert mappings and bump last_updated on conflict."""
        # 1) Normalize columns and drop missing cusips
        clean = pd.DataFrame({
            'cusip':  df.get('cusip',  df.get('CUSIP')),
            'isin':   df.get('isin',   df.get('ISIN')),
            'ticker': df.get('ticker', df.get('Ticker'))
        }).dropna(subset=['cusip'])

        # 2) Convert to list-of-dicts
        records = clean.to_dict(orient='records')

        # 3) Build a single INSERT … ON CONFLICT DO UPDATE
        stmt = insert(self.mappings_table).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=['cusip'],
            set_={
                'isin':         stmt.excluded.isin,
                'ticker':       stmt.excluded.ticker,
                'last_updated': func.current_timestamp()
            }
        )

        # 4) Execute in one go
        with self.engine.begin() as conn:
            conn.execute(stmt)
    
    def _query_with_preserve_input(self, identifiers: List[str], from_col: str, to_col: str) -> pd.DataFrame:
        """Helper method that preserves input order and includes NaNs for missing mappings"""
        # Query database (only non-null results)
        stmt = select(getattr(self.mappings_table.c, from_col), 
                     getattr(self.mappings_table.c, to_col)).where(
            getattr(self.mappings_table.c, from_col).in_(identifiers) & 
            getattr(self.mappings_table.c, to_col).is_not(None)
        )
        
        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            db_result = pd.DataFrame(result.fetchall(), columns=[from_col, to_col])
        
        # Create DataFrame with all input identifiers
        input_df = pd.DataFrame({from_col: identifiers})
        
        # Left join to preserve all inputs and add NaNs for missing
        final_result = input_df.merge(db_result, on=from_col, how='left')
        
        return final_result
    
    def _query_with_mixed_input(self, identifiers: List[str], to_col: str) -> pd.DataFrame:
        stmt = (
            select(self.mappings_table)
            .where(
                (self.mappings_table.c.cusip.in_(identifiers) |
                 self.mappings_table.c.isin.in_(identifiers) |
                 self.mappings_table.c.ticker.in_(identifiers))
                & getattr(self.mappings_table.c, to_col).is_not(None)
            )
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        df = pd.DataFrame(rows, columns=['cusip','isin','ticker','last_updated'])

        if df.empty:
            return pd.DataFrame({'identifier': identifiers, to_col: [pd.NA]*len(identifiers)})
        
        # this builds a DataFrame of say cusip: cusip, cusip: isin, and cusip: ticker
        # so the identifiers are all possible identifiers used to be merged with the requested identifiers
        melted = (
            df
            .melt(
                id_vars=[to_col],
                value_vars=['cusip','isin','ticker'],
                var_name='id_type',
                value_name='identifier'
            )
            .dropna(subset=['identifier'])
        )

        mapping_df = (
            melted
            .loc[:, ['identifier', to_col]]
            .drop_duplicates('identifier', keep='first')
        )

        # 4) Re-attach to your original list to preserve order
        out = pd.DataFrame({'identifier': identifiers})
        out = out.merge(mapping_df, on='identifier', how='left')

        return out
    
    def mixed_to_cusip(self, identifiers: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        
        return self._query_with_mixed_input(identifiers, 'cusip')
    
    def _check_existence(self, identifiers: List[str], column: str) -> pd.DataFrame:
        """Helper method to check if identifiers exist in database"""
        # Query for existence (don't care about other columns being null)
        stmt = select(getattr(self.mappings_table.c, column)).where(
            getattr(self.mappings_table.c, column).in_(identifiers)
        )
        
        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            existing = pd.DataFrame(result.fetchall(), columns=[column])
        
        # Create DataFrame with all input identifiers
        input_df = pd.DataFrame({column: identifiers})
        
        # Add exists column
        input_df['exists'] = input_df[column].isin(existing[column])
        
        return input_df
    
    # Existence checking methods
    def check_cusips_exist(self, cusips: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(cusips, str):
            cusips = [cusips]
        
        return self._check_existence(cusips, 'cusip')
    
    def check_isins_exist(self, isins: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(isins, str):
            isins = [isins]
        
        return self._check_existence(isins, 'isin')
    
    def check_tickers_exist(self, tickers: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        
        return self._check_existence(tickers, 'ticker')
    
    def check_mixed_exist(self, identifiers: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        
        # Check existence across all three columns
        stmt = (
            select(self.mappings_table.c.cusip, self.mappings_table.c.isin, self.mappings_table.c.ticker)
            .where(
                (self.mappings_table.c.cusip.in_(identifiers) |
                 self.mappings_table.c.isin.in_(identifiers) |
                 self.mappings_table.c.ticker.in_(identifiers))
            )
        )
        
        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            existing = pd.DataFrame(result.fetchall(), columns=['cusip', 'isin', 'ticker'])
        
        # Create DataFrame with all input identifiers
        input_df = pd.DataFrame({'identifier': identifiers})
        
        # Add exists column based on any of the three columns
        input_df['exists'] = input_df['identifier'].isin(existing[['cusip', 'isin', 'ticker']].values.flatten())
        
        return input_df
    
    # Convenience methods for boolean results
    def cusips_exist(self, cusips: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Return boolean(s) for existence"""
        result = self.check_cusips_exist(cusips)
        if isinstance(cusips, str):
            return result['exists'].iloc[0]
        return result['exists'].tolist()
    
    def cusips_not_exist(self, cusips: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Return boolean(s) for non-existence"""
        result = self.check_cusips_exist(cusips)
        if isinstance(cusips, str):
            return not result['exists'].iloc[0]
        return [not exists for exists in result['exists'].tolist()]
    
    def isins_exist(self, isins: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Return boolean(s) for existence"""
        result = self.check_isins_exist(isins)
        if isinstance(isins, str):
            return result['exists'].iloc[0]
        return result['exists'].tolist()
    
    def tickers_exist(self, tickers: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Return boolean(s) for existence"""
        result = self.check_tickers_exist(tickers)
        if isinstance(tickers, str):
            return result['exists'].iloc[0]
        return result['exists'].tolist()
    
    # Original mapping methods
    def cusip_to_ticker(self, cusips: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(cusips, str):
            cusips = [cusips]
        
        return self._query_with_preserve_input(cusips, 'cusip', 'ticker')
    
    def cusip_to_isin(self, cusips: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(cusips, str):
            cusips = [cusips]
        
        return self._query_with_preserve_input(cusips, 'cusip', 'isin')
    
    def ticker_to_cusip(self, tickers: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        
        return self._query_with_preserve_input(tickers, 'ticker', 'cusip')
    
    def ticker_to_isin(self, tickers: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        
        return self._query_with_preserve_input(tickers, 'ticker', 'isin')
    
    def isin_to_cusip(self, isins: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(isins, str):
            isins = [isins]
        
        return self._query_with_preserve_input(isins, 'isin', 'cusip')
    
    def isin_to_ticker(self, isins: Union[str, List[str]]) -> pd.DataFrame:
        if isinstance(isins, str):
            isins = [isins]
        
        return self._query_with_preserve_input(isins, 'isin', 'ticker')
    
    def get_all_mappings(self) -> pd.DataFrame:
        stmt = select(self.mappings_table)
        
        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            return pd.DataFrame(result.fetchall(), columns=['cusip', 'isin', 'ticker', 'last_updated'])