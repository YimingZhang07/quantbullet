import unittest
import tempfile
import shutil
import pandas as pd
from pathlib import Path

from quantbullet.utils import cache_variables, load_cache_variables

class TestCacheUtils(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

        # Sample variables
        self.df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        self.data = {'key': 'value', 'num': 42}

    def tearDown(self):
        # Clean up the temporary directory after test
        shutil.rmtree(self.test_dir)

    def test_cache_variables_with_timestamp(self):
        result_path = cache_variables(self.test_dir, df=self.df, data=self.data)
        self.assertTrue(Path(result_path).exists())
        self.assertTrue(Path(result_path, 'df.pkl').exists())
        self.assertTrue(Path(result_path, 'data.pkl').exists())

    def test_cache_variables_with_custom_subfolder(self):
        result_path = cache_variables(self.test_dir, subfolder="test_subfolder", df=self.df)
        expected_path = Path(self.test_dir, "test_subfolder")
        self.assertEqual(result_path, str(expected_path))
        self.assertTrue(expected_path.exists())
        self.assertTrue(Path(expected_path, 'df.pkl').exists())

    def test_load_cache_variables_single(self):
        subfolder = "load_test"
        path = cache_variables(self.test_dir, subfolder=subfolder, df=self.df)
        result = load_cache_variables(path, 'df')
        pd.testing.assert_frame_equal(result['df'], self.df)

    def test_load_cache_variables_multiple(self):
        path = cache_variables(self.test_dir, df=self.df, data=self.data)
        result = load_cache_variables(path, 'df', 'data')
        pd.testing.assert_frame_equal(result['df'], self.df)
        self.assertEqual(result['data'], self.data)

    def test_load_nonexistent_variable(self):
        path = cache_variables(self.test_dir, df=self.df)
        with self.assertRaises(FileNotFoundError):
            load_cache_variables(path, 'nonexistent')

    def test_save_to_nonexistent_base_dir(self):
        bad_dir = Path(self.test_dir, 'does_not_exist')
        with self.assertRaises(ValueError):
            cache_variables(bad_dir, df=self.df)
