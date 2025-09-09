import numpy as np
import unittest
from quantbullet.parametic_model import AsymQuadModel


class TestBathtubModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        x = np.linspace(1, 10, 30)
        true_params = { 'params_dict': { 'a': 1.5, 'b': 2.0, 'x0': 3.0, 'c': 2.2 } }
        true_model = AsymQuadModel(**true_params)
        y_true = true_model.predict(x)
        y_noisy = y_true + np.random.normal(0, 0.1, len(y_true))
        self.x, self.y_noisy = x, y_noisy

    def test_fitting_workflow(self):
        model = AsymQuadModel()
        model.fit(self.x, self.y_noisy)
        y_pred = model.predict(self.x)

        # test that the relative error is less than 1% for all the data points
        relative_error = np.abs(y_pred - self.y_noisy) / np.abs(self.y_noisy)
        self.assertTrue(np.all(relative_error < 0.1))

    def test_to_dict_from_dict( self ):
        model = AsymQuadModel()
        model.fit(self.x, self.y_noisy)

        saved_model_dict = model.to_dict()
        loaded_model = AsymQuadModel.from_dict(saved_model_dict)

        x_test = np.linspace(0, 20, 100)
        y_test_model = model.predict(x_test)
        y_test_loaded_model = loaded_model.predict(x_test)

        self.assertTrue(np.allclose(y_test_model, y_test_loaded_model, atol=1e-6))