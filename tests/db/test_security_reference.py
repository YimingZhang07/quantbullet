import unittest
from quantbullet.db.security_reference_sqlite import SecurityReferenceCache
from sqlalchemy import create_engine
from unittest.mock import patch

class TestSecurityReference(unittest.TestCase):
    def setUp(self):
        engine = create_engine('sqlite:///:memory:', future=True)
        self.cache = SecurityReferenceCache(cache_dir="test_cache", engine=engine)

        self.normal_mapping = {
            'cusip' : ['ABC12345678', 'DEF23456789'],
            'isin'  : ['USABC1234567', 'USDEF2345678'],
            'ticker': ['ABC', 'DEF'],
        }

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