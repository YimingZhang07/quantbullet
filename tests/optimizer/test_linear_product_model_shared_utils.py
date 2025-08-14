from quantbullet.optimizer.linear_product_shared import init_betas_by_response_mean

import unittest
import numpy as np

class TestLinearProductShared(unittest.TestCase):
    def test_init_betas_by_response_mean(self):
        X = np.random.rand(100, 5)  # Example feature matrix
        target_mean = 10.0  # Desired target mean
        betas = init_betas_by_response_mean(X, target_mean)
        response_mean = np.mean( X @ betas )
        assert np.isclose(response_mean, target_mean, atol=1e-2)
        self.assertIsInstance(betas , np.ndarray)