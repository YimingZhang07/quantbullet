import unittest
import numpy as np
import pandas as pd
from quantbullet.dfutils.label import get_bins_and_labels
from quantbullet.linear_product_model import LinearProductRegressorBCD, LinearProductRegressorScipy
from quantbullet.preprocessing import FlatRampTransformer
from sklearn.metrics import mean_squared_error

class TestLinearProductModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = int( 10e4 )
        df = pd.DataFrame({
            'x1': 10 + 3 * np.random.randn(n_samples),
            'x2': 20 + 3 * np.random.randn(n_samples),
            'x3': np.random.choice([0, 1, 2], size=n_samples, p=[0.25, 0.25, 0.5])
        })

        df['x3'] = df['x3'].astype('category')

        df['y'] = np.cos(df['x1']) + np.sin(df['x2']) + \
            np.random.randn(n_samples) * 0.5 + \
            np.random.normal(loc = df['x3'].cat.codes, scale=0.5) + 10

        x1_trans = FlatRampTransformer(
            knots = list( np.arange( 4, 16, 1 ) ),
            include_bias=True
        )

        x2_trans = FlatRampTransformer(
            knots = list( np.arange( 14, 26, 1 ) ),
            include_bias=True
        )

        train_df = np.concatenate([
            x1_trans.fit_transform(df['x1']),
            x2_trans.fit_transform(df['x2']),
            pd.get_dummies(df['x3'], prefix='x3') * 1
        ], axis=1)

        train_df = pd.DataFrame(train_df, columns = x1_trans.get_feature_names_out().tolist() 
                                + x2_trans.get_feature_names_out().tolist()
                                + [ 'x3_' + str(cat) for cat in df['x3'].cat.categories.tolist() ])
        
        x1_bins, x1_labels = get_bins_and_labels(cutoffs=list( np.arange(4, 16, 1 ) ) )
        x2_bins, x2_labels = get_bins_and_labels(cutoffs=list( np.arange(14, 26, 1 ) ) )

        df['x1_bins'] = pd.cut( df['x1'], bins=x1_bins, labels=x1_labels )
        df['x2_bins'] = pd.cut( df['x2'], bins=x2_bins, labels=x2_labels )

        feature_groups = {'x1': x1_trans.get_feature_names_out().tolist(), 
                        'x2': x2_trans.get_feature_names_out().tolist(),
                        'x3': [ 'x3_' + str(cat) for cat in df['x3'].cat.categories.tolist() ]}
        
        self.df = df
        self.train_df = train_df
        self.feature_groups = feature_groups

    def test_LinearProductModelOLS(self):
        df = self.df
        train_df = self.train_df
        feature_groups = self.feature_groups

        lprm_ols = LinearProductRegressorBCD()
        lprm_ols.fit( train_df, df['y'], feature_groups=feature_groups, n_iterations=10, early_stopping_rounds=5 )
        df['model_pred_BCD'] = lprm_ols.predict(train_df)

        # check the mean squared error
        mse = mean_squared_error(df['y'], df['model_pred_BCD'])
        self.assertTrue( np.isclose(mse, 0.52, atol=0.01), f"MSE is {mse}, expected around 0.52" )

    def test_LinearProductModelScipy(self):
        df = self.df
        train_df = self.train_df
        feature_groups = self.feature_groups

        lpm_scipy = LinearProductRegressorScipy( xtol=1e-12, gtol=1e-12, ftol=1e-12 )
        lpm_scipy.fit( X=train_df, y=df['y'], feature_groups=feature_groups, verbose=1 )
        df['model_predict_scipy'] = lpm_scipy.predict(train_df)

        # check the mean squared error
        mse = mean_squared_error(df['y'], df['model_predict_scipy'])
        self.assertTrue( np.isclose(mse, 0.52, atol=0.01), f"MSE is {mse}, expected around 0.52" )