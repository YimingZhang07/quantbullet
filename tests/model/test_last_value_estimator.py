import unittest
import pandas as pd
from quantbullet.model.benchmark_estimators import LastValueEstimator

class TestLastValueEstimator(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'x1': [1, 2, 3, 4, 5],
            'x2': [5, 4, 3, 2, 1],
            'y': [10, 20, 30, 40, 50]
        })
        self.X = self.df[['timestamp', 'x1', 'x2']]
        self.y = self.df['y']
        
    def test_fit(self):
        model = LastValueEstimator(reference_column='timestamp')
        model.fit(self.X, self.y)
        self.assertTrue(hasattr(model, 'last_value_'))
        self.assertEqual(model.last_value_, 50)
        X_test = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-06', periods=5, freq='D'),
            'x1': [6, 7, 8, 9, 10],
            'x2': [0, -1, -2, -3, -4]
        })
        predictions = model.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
        self.assertTrue(all(predictions == 50))