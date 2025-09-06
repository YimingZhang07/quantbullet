import numpy as np
import unittest
from quantbullet.parametic_model import BathtubModel


class TestBathtubModel(unittest.TestCase):
    def test_fitting_workflow(self):
        """Test complete fit and predict workflow with sample data."""
        # Generate sample data with noise
        np.random.seed(42)
        x = np.linspace(0.1, 10, 30)
        true_params = {'lam': 1.5, 'theta1': 2.0, 'k1': 1.8, 'theta2': 3.0, 'k2': 2.2}
        
        # Create true model and generate data
        true_model = BathtubModel(true_params)
        y_true = true_model.predict(x)
        y_noisy = y_true + np.random.normal(0, 0.1, len(y_true))
        
        # Fit new model
        model = BathtubModel()
        model.fit(x, y_noisy)
        
        # Test prediction
        y_pred = model.predict(x)
        
        # Basic checks
        self.assertEqual(len(y_pred), len(x))
        self.assertTrue(np.all(np.isfinite(y_pred)))
        self.assertTrue(np.all(y_pred > 0))

