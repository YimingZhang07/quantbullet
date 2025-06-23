import unittest
import pandas as pd
from quantbullet.db.security_reference_sqlite import SecurityReferenceCache
from sqlalchemy import create_engine
from unittest.mock import patch

class TestSecurityReference(unittest.TestCase):
    def setUp(self):
        engine = create_engine('sqlite:///:memory:', future=True)
        self.cache = SecurityReferenceCache(cache_dir="test_cache", engine=engine)

        self.normal_mapping = pd.DataFrame({
            'cusip' : ['ABC12345678', 'DEF23456789'],
            'isin'  : ['USABC1234567', 'USDEF2345678'],
            'ticker': ['ABC', 'DEF'],
        })

        self.invalid_identifiers = [ 'XYZ123', 'XYZ456' ]

    def test_add_and_get_mapping(self):
        self.cache.add_mappings( self.normal_mapping )
        result = self.cache.get_all_mappings()
        self.assertEqual(len(result), 2)
        self.assertTrue( 'ABC12345678' in result['Cusip'].to_list() )

    def test_query_mixed_to_cusips( self ):
        self.cache.add_mappings( self.normal_mapping )
        result = self.cache.mixed_to_cusip(['USABC1234567', 'DEF'])
        self.assertEqual(len(result), 2)
        self.assertTrue( result['Cusip'].isna().sum() == 0 )

    def test_check_mixed_exist( self ):
        self.cache.add_mappings( self.normal_mapping )
        identifiers_to_check = [ 'ABC12345678', 'DEF', 'XYZ' ]
        result = self.cache.check_mixed_exist(identifiers_to_check)
        self.assertEqual(len(result), 3)
        self.assertTrue(result['Exists'].sum() == 2)

    def test_query_duplicates(self):
        self.cache.add_mappings(self.normal_mapping)
        result = self.cache.cusip_to_isin(['ABC12345678', 'ABC12345678'])
        self.assertEqual(len(result), 2)
        self.assertTrue(result['ISIN'].unique().size == 1)

        # in the below case, we have a duplicate ticker, and a non-exist ticker
        # the unique result will include the nan values as well so the size will be 3
        result = self.cache.mixed_to_cusip(['USABC1234567', 'DEF', 'DEF', 'XYZ'])
        self.assertEqual(len(result), 4)
        self.assertTrue(result['Cusip'].isna().sum() == 1)
        self.assertTrue(result['Cusip'].unique().size == 3)

    def test_add_invalid_identifier(self):
        self.cache.add_invalid_identifiers(self.invalid_identifiers)
        result = self.cache.get_invalid_identifiers()
        self.assertEqual(len(result), 2)
        self.assertTrue('XYZ123' in result['Identifier'].to_list())

    def test_check_identifier_invalid(self):
        self.cache.add_invalid_identifiers(self.invalid_identifiers)
        result = self.cache.check_identifier_invalid(['XYZ123', 'ABC123'])
        self.assertIsInstance( result, pd.DataFrame )
        self.assertEqual( len( result ), 2 )
        self.assertTrue( result['Exists'].any() )
        self.assertFalse( result['Exists'].all() )

    def test_mixed_to_cusip_with_invalid(self):
        self.cache.add_mappings(self.normal_mapping)
        self.cache.add_invalid_identifiers(self.invalid_identifiers)
        result = self.cache.mixed_to_cusip(['USABC1234567', 'XYZ123'])
        self.assertEqual(len(result), 2)
        self.assertTrue(result['Cusip'].isna().sum() == 1)
        self.assertTrue( 'Success' in result['Status'].values )
        self.assertTrue( 'Invalid' in result['Status'].values )