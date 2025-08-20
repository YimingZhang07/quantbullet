import unittest
import numpy as np
import statsmodels.api as sm
from scipy.optimize import approx_fprime
from quantbullet.log_config import setup_logger
from quantbullet.linear_product_model.utils import (
    estimate_ols_beta_se, 
    estimate_logistic_beta_se,
    numerical_hessian,
    linear_clipped_sum_cross_entropy_loss,
    estimate_linear_cross_entropy_beta_se,
    init_betas_by_response_mean
)

logger = setup_logger(__name__)

class Test_OLS_MSE_Error_Model(unittest.TestCase):
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
        se_mine = estimate_ols_beta_se(X, y, beta_hat)

        # Compare to statsmodels
        model = sm.OLS(y, X).fit()
        se_sm = model.bse
        
        assert np.allclose(se_mine, se_sm, atol=1e-6)
        
    def test_numerical_beta_se(self):
        from scipy.optimize import approx_fprime

        def mse_loss(beta, X, y):
            residuals = y - X @ beta
            return 0.5 * np.sum(residuals**2)
        
        X, y, beta_hat = self.X, self.y, self.beta_hat

        # My SE estimate
        se_mine = estimate_ols_beta_se(X, y, beta_hat)
        n, p = X.shape
        
        # Hessian-based SE estimate
        H = numerical_hessian(mse_loss, beta_hat, 1e-5, X, y)
        sigma2 = np.sum((y - X @ beta_hat) ** 2) / (n - p)
        cov_beta = sigma2 * np.linalg.inv(H)   # since Hessian = X^T X
        se_hess = np.sqrt(np.diag(cov_beta))
        
        logger.info(f"SE mine: {se_mine}")
        logger.info(f"SE hessian: {se_hess}")

        assert np.allclose(se_mine, se_hess, atol=1e-5)

class Test_Logit_Cross_Entropy_Error_Model(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n, p = 200, 3
        X = np.random.randn(n, p)
        beta_true = np.array([1.0, -2.0, 0.5])
        linpred = X @ beta_true
        p_true = 1 / (1 + np.exp(-linpred))
        y = np.random.binomial(1, p_true)

        # Fit logistic regression with statsmodels to get MLE beta
        model = sm.Logit(y, X).fit(disp=False)
        beta_hat = model.params

        self.X, self.y, self.beta_hat, self.model = X, y, beta_hat, model

    def test_analytical_beta_se(self):
        """Compare my SE against statsmodels.Logit."""
        se_mine = estimate_logistic_beta_se(self.X, self.beta_hat)
        se_sm = self.model.bse

        assert np.allclose(se_mine, se_sm, atol=1e-6)

    def test_numerical_beta_se(self):
        """Compare my SE against numerical Hessian inversion."""
        def neg_loglik(beta, X, y):
            xb = X @ beta
            p = 1 / (1 + np.exp(-xb))
            # Negative log-likelihood
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        H = numerical_hessian(neg_loglik, self.beta_hat, 1e-5, self.X, self.y)
        cov_beta = np.linalg.inv(H)  # Fisher info inverse
        se_hess = np.sqrt(np.diag(cov_beta))

        se_mine = estimate_logistic_beta_se(self.X, self.beta_hat)

        assert np.allclose(se_mine, se_hess, atol=1e-4)

class Test_Linear_Cross_Entropy_Error_Model(unittest.TestCase):
    def setUp(self):
        """To run a fake example and test the implementation, just remember that the data geneartion has to
        at least align with the model form.

        We have to make an example where prob is linear to X @ beta;
        """
        np.random.seed(42)
        n, p = 1000, 3
        X = np.random.randn(n, p) + 10
        beta_true = init_betas_by_response_mean(X, 0.2)
        probs = X @ beta_true
        probs = np.clip(probs, 0, 1)
        y = np.random.binomial(1, probs)

        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        self.X, self.y, self.beta_hat = X, y, beta_hat

    def test_numerical_beta_se(self):

        def linear_clipped_cross_entropy_loss(beta, X, y):
            return linear_clipped_sum_cross_entropy_loss(X, y, beta, eps=1e-6)

        H = numerical_hessian( linear_clipped_cross_entropy_loss, self.beta_hat, 1e-5, self.X, self.y )
        cov_beta = np.linalg.inv(H)  # Fisher info inverse
        se_hess = np.sqrt(np.diag(cov_beta))
        se_mine = estimate_linear_cross_entropy_beta_se(self.X, self.y, self.beta_hat, eps=1e-6)

        # compare the ratios
        ratio = se_mine / se_hess
        # ratio should be between 0.8 and 1.2
        self.assertTrue(np.all(ratio > 0.8) and np.all(ratio < 1.2), "Ratios should be between 0.8 and 1.2")

    def test_mse_cross_entropy_differences( self ):
        """
        This compares if we falsely assume that MSE as the loss function instead of cross-entropy.
        We will underestimate the uncertainty (standard errors) of the parameter estimates.
        """

        se_linear_cross_entropy = estimate_linear_cross_entropy_beta_se(self.X, self.y, self.beta_hat, eps=1e-6)
        se_mse = estimate_ols_beta_se(self.X, self.y, self.beta_hat)

        self.assertTrue(np.all(se_linear_cross_entropy > se_mse), "Linear cross-entropy SE should be greater than MSE SE")