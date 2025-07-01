import unittest
import numpy as np
from quantbullet.model import ConstrainedLinearRegressor

class TestConstrainedLinearRegression(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_monotonic_constraints(self):
        """
        This test checks that for a positive relationship, if we set a constraint to make the coefficient negative,
        the model should train with a coefficient of zero.
        """
        # Test with constraints that should enforce monotonicity
        # make x a one dimensional array that is strictly increasing
        x = [[1], [2], [3], [4], [5]]
        y = [6, 7, 8, 9, 10]
        model = ConstrainedLinearRegressor(coef_constraints=[ "-" ], fit_intercept=True)
        model.fit(x, y)
        self.assertTrue(model.coef_[0] == 0, "Coefficient should be zero.")
        
    def test_2d_constraints(self):
        """
        This test should have an increasing x1, and a decreasing x2. and an increasing y.
        The model should learn a positive coefficient for x1 and a negative coefficient for x2.
        """
        x = np.random.rand(100, 2) * 10
        y = 2 * x[:, 0] - 3 * x[:, 1] + np.random.normal(0, 2, 100)
        model = ConstrainedLinearRegressor(coef_constraints=["+", "-"], fit_intercept=True)
        model.fit(x, y)
        self.assertTrue(model.coef_[0] > 0, "Coefficient for x1 should be positive.")
        self.assertTrue(model.coef_[1] < 0, "Coefficient for x2 should be negative.")