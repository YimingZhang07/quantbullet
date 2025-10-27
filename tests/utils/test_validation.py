import unittest
import numpy as np
import pandas as pd

from quantbullet.utils.validation import is_nan_inf_scalar

class TestValidationUtils(unittest.TestCase):
    def test_is_nan_inf_scalar(self):
        self.assertTrue(is_nan_inf_scalar(np.nan))
        self.assertTrue(is_nan_inf_scalar(float('nan')))
        self.assertTrue(is_nan_inf_scalar(np.inf))
        self.assertTrue(is_nan_inf_scalar(-np.inf))
        self.assertTrue(is_nan_inf_scalar(None))
        self.assertTrue(is_nan_inf_scalar(pd.NA))

        self.assertFalse(is_nan_inf_scalar(0))
        self.assertFalse(is_nan_inf_scalar(1.5))
        self.assertFalse(is_nan_inf_scalar("string"))
        self.assertFalse(is_nan_inf_scalar([]))