from quantbullet.linear_product_model.utils import init_betas_by_response_mean
from quantbullet.linear_product_model.utils import estiamte_ols_beta_se
from quantbullet.log_config import setup_logger
import statsmodels.api as sm

import unittest
import numpy as np

class TestLinearProductModelUtils(unittest.TestCase):
    def test_init_betas_by_response_mean(self):
        X = np.random.rand(100, 5)  # Example feature matrix
        target_mean = 10.0  # Desired target mean
        betas = init_betas_by_response_mean(X, target_mean)
        response_mean = np.mean( X @ betas )
        assert np.isclose(response_mean, target_mean, atol=1e-2)
        self.assertIsInstance(betas , np.ndarray)
