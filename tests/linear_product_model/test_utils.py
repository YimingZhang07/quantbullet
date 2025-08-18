from quantbullet.linear_product_model.utils import init_betas_by_response_mean
from quantbullet.linear_product_model.utils import estiamte_ols_beta_se
from quantbullet.log_config import setup_logger
import statsmodels.api as sm

import unittest
import numpy as np

logger = setup_logger(__name__)

class TestLinearProductModelUtils(unittest.TestCase):
    def test_init_betas_by_response_mean(self):
        X = np.random.rand(100, 5)  # Example feature matrix
        target_mean = 10.0  # Desired target mean
        betas = init_betas_by_response_mean(X, target_mean)
        response_mean = np.mean( X @ betas )
        assert np.isclose(response_mean, target_mean, atol=1e-2)
        self.assertIsInstance(betas , np.ndarray)
        
class TestErrorModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        beta_true = np.array([1.0, -2.0, 0.5])
        y = X @ beta_true + np.random.randn(n) * 0.5

        # Fit OLS beta
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        
        self.X, self.y, self.beta_hat = X, y, beta_hat

    def test_analytical_beta_se(self):
        X, y, beta_hat = self.X, self.y, self.beta_hat

        # Fit OLS beta
        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        se_mine = estiamte_ols_beta_se(X, y, beta_hat)

        # Compare to statsmodels
        model = sm.OLS(y, X).fit()
        se_sm = model.bse
        
        assert np.allclose(se_mine, se_sm, atol=1e-6)
        
    def test_numerical_beta_se(self):
        from scipy.optimize import approx_fprime

        def mse_loss(beta, X, y):
            residuals = y - X @ beta
            return 0.5 * np.sum(residuals**2)

        def numerical_hessian(fun, beta, eps=1e-5, *args):
            n = len(beta)
            H = np.zeros((n, n))
            for i in range(n):
                beta_up = beta.copy(); beta_up[i] += eps
                beta_dn = beta.copy(); beta_dn[i] -= eps
                grad_up = approx_fprime(beta_up, fun, eps, *args)
                grad_dn = approx_fprime(beta_dn, fun, eps, *args)
                H[:, i] = (grad_up - grad_dn) / (2 * eps)
            return H
        
        X, y, beta_hat = self.X, self.y, self.beta_hat

        # My SE estimate
        se_mine = estiamte_ols_beta_se(X, y, beta_hat)
        n, p = X.shape
        
        # Hessian-based SE estimate
        H = numerical_hessian(mse_loss, beta_hat, 1e-5, X, y)
        sigma2 = np.sum((y - X @ beta_hat) ** 2) / (n - p)
        cov_beta = sigma2 * np.linalg.inv(H)   # since Hessian = X^T X
        se_hess = np.sqrt(np.diag(cov_beta))
        
        logger.info(f"SE mine: {se_mine}")
        logger.info(f"SE hessian: {se_hess}")

        assert np.allclose(se_mine, se_hess, atol=1e-5)