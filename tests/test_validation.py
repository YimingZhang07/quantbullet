from quantbullet.utils.validation import are_only_values_in_series
import pandas as pd
import unittest


class TestValidationFunctions(unittest.TestCase):
    def test_are_only_values_in_series(self):
        series = pd.Series([1, 2, 3, 4, 5])
        self.assertTrue(are_only_values_in_series(series, [1, 2, 3, 4, 5]))
        self.assertFalse(are_only_values_in_series(series, [1, 2, 3, 4, 6]))
        self.assertFalse(are_only_values_in_series(series, [1, 2, 3, 4]))
