import unittest
import os
import shutil
import time
from datetime import date
from quantbullet.utils.decorators import normalize_date_args, disk_cache

@normalize_date_args("as_of_date", "settle_date")
def func(as_of_date, settle_date=None):
    return as_of_date, settle_date

@disk_cache("./tests/_cache_dir")
def cached_func(x, y):
    time.sleep(2)  # Simulate a time-consuming computation
    return x + y

class TestDecorators(unittest.TestCase):
    def test_normalize_date_args(self):
        # Test with date objects
        as_of_date = date(2023, 10, 1)
        settle_date = date(2023, 10, 2)
        result = func(as_of_date, settle_date)
        self.assertEqual(result, (date(2023, 10, 1), date(2023, 10, 2)))

        # Test with string dates
        as_of_date_str = "20231001"
        settle_date_str = "20231002"
        result = func(as_of_date_str, settle_date_str)
        self.assertEqual(result, (date(2023, 10, 1), date(2023, 10, 2)))

        # Test with string dates
        as_of_date_str = "2023-10-01"
        settle_date_str = "2023-10-02"
        result = func(as_of_date_str, settle_date_str)
        self.assertEqual(result, (date(2023, 10, 1), date(2023, 10, 2)))

        # Test with mixed types
        result = func(as_of_date_str, settle_date)
        self.assertEqual(result, (date(2023, 10, 1), date(2023, 10, 2)))

class TestDiskCache(unittest.TestCase):

    def setUp(self):
        self.cache_dir = "./tests/_cache_dir"
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_basic_caching(self):
        # First call should compute and cache
        result1 = cached_func(3, 4)
        self.assertEqual(result1, 7)

        # Cache file should exist
        files = os.listdir(self.cache_dir)
        self.assertTrue(any(f.endswith(".pkl") for f in files))
        self.assertTrue(any(f.endswith(".json") for f in files))

        # Second call should load from cache (no recompute)
        result2 = cached_func(3, 4)
        self.assertEqual(result2, 7)

        # Different args → new cache entry
        result3 = cached_func(5, 6)
        self.assertEqual(result3, 11)
        files_after = os.listdir(self.cache_dir)
        self.assertGreater(len(files_after), len(files))

    def test_force_recache(self):
        # First call
        result1 = cached_func(10, 20)
        self.assertEqual(result1, 30)

        # Force recache
        result2 = cached_func(10, 20, force_recache=True)
        self.assertEqual(result2, 30)

    def test_expire_days(self):
        # First call → cached
        result1 = cached_func(1, 2)
        self.assertEqual(result1, 3)

        # Expire immediately → should recompute
        result2 = cached_func(1, 2, expire_days=0)
        self.assertEqual(result2, 3)

    def test_cache_speedup(self):
        # First call: should be slow
        start = time.time()
        result1 = cached_func(3, 4)
        elapsed_first = time.time() - start
        self.assertEqual(result1, 7)

        # Second call: should be much faster (cache hit)
        start = time.time()
        result2 = cached_func(3, 4)
        elapsed_second = time.time() - start
        self.assertEqual(result2, 7)

        # Verify second call is faster (at least 10x faster or <0.1s)
        self.assertTrue(elapsed_second < elapsed_first / 10 or elapsed_second < 0.1,
                        f"Cache not used, elapsed_first={elapsed_first}, elapsed_second={elapsed_second}")

    def test_force_recache_speed(self):
        # Warm up cache
        cached_func(5, 6)

        # Call with force_recache: should take ~2s again
        start = time.time()
        result = cached_func(5, 6, force_recache=True)
        elapsed = time.time() - start
        self.assertEqual(result, 11)
        self.assertTrue(elapsed >= 2, "force_recache should recompute, but was too fast")