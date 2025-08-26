import unittest
import numpy as np
from quantbullet.linear_product_model.utils import init_betas_by_response_mean

def qr_solve(X, y):
    """Solve least squares using QR decomposition."""
    Q, R = np.linalg.qr(X)
    return np.linalg.solve(R, Q.T @ y)

def closed_form_solve(X, y):
    """Solve least squares using (X^T X)^-1 X^T y."""
    XtX_inv = np.linalg.inv(X.T @ X)
    return XtX_inv @ X.T @ y

class TestLinearProductModelUtils(unittest.TestCase):
    def test_init_betas_by_response_mean(self):
        X = np.random.rand(100, 5)  # Example feature matrix
        target_mean = 10.0  # Desired target mean
        betas = init_betas_by_response_mean(X, target_mean)
        response_mean = np.mean( X @ betas )
        assert np.isclose(response_mean, target_mean, atol=1e-2)
        self.assertIsInstance(betas , np.ndarray)

    def test_qr_vs_closed_form_small(self):
        np.random.seed(0)
        X = np.random.randn(1_000, 3)
        y = np.random.randn(1_000)

        beta_qr = qr_solve(X, y)
        beta_cf = closed_form_solve(X, y)

        np.testing.assert_allclose(beta_qr, beta_cf, rtol=1e-10, atol=1e-12)