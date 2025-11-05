import unittest
from quantbullet.utils.os import DiskCacheCleaner, iter_subfolders
from datetime import timedelta
from quantbullet.log_config import setup_logger

logger = setup_logger( __name__ )

class TestDiskCacheCleaner( unittest.TestCase ):
    def setUp( self ):
        self.cache_dir = None

    def tearDown(self):
        pass
    
    def test_disk_cache_cleaner( self ):
        cleaner = DiskCacheCleaner( self.cache_dir, time_limit=timedelta( days=7 ) )
        cleaner.clean( dry_run=False )

    def test_iter_subfolders(self):
        folders = list( iter_subfolders( self.cache_dir, include_parent=True ) )
        self.assertTrue( len( folders ) > 0 )

    def test_disk_cache_cleaner_all_subfolders( self ):
        for folder in iter_subfolders( self.cache_dir, include_parent=True ):
            logger.info( f"Cleaning folder: {folder}" )
            cleaner = DiskCacheCleaner( folder, time_limit=timedelta( days=7 ) )
            cleaner.clean( dry_run=True )