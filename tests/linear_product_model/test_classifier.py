import unittest
import numpy as np
import pandas as pd
from quantbullet.preprocessing import FlatRampTransformer
from quantbullet.linear_product_model import LinearProductClassifierScipy

from quantbullet.utils.decorators import log_runtime

class TestLinearProductClassifierSingleFeatureGroup(unittest.TestCase):
    def setUp(self):
        # generate synthetic data
        np.random.seed(42)
        n_samples = 100_000
        x1 = np.random.uniform(0, 4, n_samples)
        y = ( x1 - 2 ) ** 2 + np.random.normal(0, 1, n_samples) + 10
        df = pd.DataFrame({'x1': x1, 'y': y})
        
        # create piecewise linear features for x1
        x1_trans = FlatRampTransformer(
            knots = [0.5, 1, 2, 3, 3.5],
            include_bias=True
        )

        train_df = np.concatenate([
            x1_trans.fit_transform(df['x1']),
        ], axis=1)

        train_df = pd.DataFrame(train_df, columns = x1_trans.get_feature_names_out().tolist() )
        
        # generate binary target variable
        probs = 1 / (1 + np.exp(-(df['y'] - 16)))
        df['binary_y'] = np.random.binomial(1, probs)
        
        feature_groups = {'x1': x1_trans.get_feature_names_out().tolist()}
        
        self.df = df
        self.train_df = train_df
        self.feature_groups = feature_groups
        
    def test_fit(self):
        df = self.df
        train_df = self.train_df
        feature_groups = self.feature_groups
        
        lpc_scipy = LinearProductClassifierScipy(ftol=1e-6, gtol=1e-6, eps=1e-3)
        lpc_scipy.fit( train_df, df['binary_y'], feature_groups=feature_groups, use_jacobian=True )
        preds = lpc_scipy.predict(train_df)
        
        self.assertTrue( np.isclose(preds.mean(), df['binary_y'].mean(), atol=1e-4) )