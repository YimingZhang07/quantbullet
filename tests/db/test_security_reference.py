import unittest
from quantbullet.db.security_reference_sqlite import SimpleMappingCache
from sqlalchemy import create_engine
from unittest.mock import patch

class TestSecurityReference(unittest.TestCase):
    def setUp(self):
        engine = create_engine('sqlite:///:memory:', future=True)
        self.cache = SimpleMappingCache(cache_dir="test_cache", engine=engine)

        self.normal_mapping = {
            'cusip' : ['ABC12345678', 'DEF23456789'],
            'isin'  : ['USABC1234567', 'USDEF2345678'],
            'ticker': ['ABC', 'DEF'],
        }

    def test_add_and_get_mapping(self):
        self.cache.add_mappings( self.normal_mapping )
        result = self.cache.get_all_mappings()
        self.assertEqual(len(result), 2)
        self.assertTrue( 'ABC12345678' in result['cusip'].to_list() )