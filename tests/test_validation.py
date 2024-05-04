import pandas as pd
import numpy as np
import unittest

from quantbullet.utils.validation import are_only_values_in_series, Consolidator

class TestValidationFunctions(unittest.TestCase):
    def test_are_only_values_in_series(self):
        series = pd.Series([1, 2, 3, 4, 5])
        self.assertTrue(are_only_values_in_series(series, [1, 2, 3, 4, 5]))
        self.assertFalse(are_only_values_in_series(series, [1, 2, 3, 4, 6]))
        self.assertFalse(are_only_values_in_series(series, [1, 2, 3, 4]))

    
class TestConsolidator(unittest.TestCase):
    def test_consolidate_to_series(self):
        series = Consolidator.consolidate_to_series([1, 2, 3, 4, 5])
        self.assertIsInstance(series, pd.Series)
        self.assertTrue(series.index.equals(pd.RangeIndex(start=0, stop=5)))
        series = Consolidator.consolidate_to_series(pd.Series([1, 2, 3, 4, 5], index=[2, 3, 5, 8, 9]))
        self.assertIsInstance(series, pd.Series)
        self.assertTrue(series.index.equals(pd.Index([2, 3, 5, 8, 9])))
