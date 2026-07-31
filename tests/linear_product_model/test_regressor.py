import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

from quantbullet.linear_product_model import (
    LinearProductRegressorBCD,
    LinearProductRegressorScipy,
    LinearProductModelToolkit,
)
from quantbullet.preprocessing import FlatRampTransformer
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
from quantbullet.model.feature import DataType, Feature, FeatureRole, FeatureSpec
from quantbullet.parametric_model import InterpolatedModel

def _setup_toolkit():
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

    preprocess_config = {
        'x1': FlatRampTransformer(
            knots = list( np.arange( 4, 16, 1 ) ),
            include_bias=True
        ),
        'x2': FlatRampTransformer(
            knots = list( np.arange( 14, 26, 1 ) ),
            include_bias=True
        ),
        'x3': OneHotEncoder( drop=None )
    }

    feature_spec = FeatureSpec(
        features = [
            Feature( name='x1', dtype=DataType.FLOAT, role=FeatureRole.MODEL_INPUT ),
            Feature( name='x2', dtype=DataType.FLOAT, role=FeatureRole.MODEL_INPUT ),
            Feature( name='x3', dtype=DataType.CATEGORY, role=FeatureRole.MODEL_INPUT ),
            Feature( name='y', dtype=DataType.FLOAT, role=FeatureRole.TARGET )
        ]
    )

    tk = LinearProductModelToolkit( feature_spec=feature_spec, preprocess_config=preprocess_config ).fit( df )

    return df, tk

def _setup_data():
    df, tk = _setup_toolkit()
    return df, tk.get_expanded_df( df ), tk.feature_groups

class TestLinearProductRegressorBCD(unittest.TestCase):
    def setUp(self):
        df, expanded_df, feature_groups = _setup_data()
        self.df = df
        self.train_df = expanded_df
        self.feature_groups = feature_groups

    def test_LinearProductRegressorBCD(self):
        df = self.df
        train_df = self.train_df
        feature_groups = self.feature_groups

        dcontainer = ProductModelDataContainer( df, train_df, response= df['y'], feature_groups=feature_groups )
        lprm_ols = LinearProductRegressorBCD()
        lprm_ols.fit( dcontainer, feature_groups=feature_groups, n_iterations=100, early_stopping_rounds=20, cache_qr_decomp=False )
        df['model_pred_BCD'] = lprm_ols.predict( dcontainer )

        # check the mean squared error; whether the model converges
        mse = mean_squared_error(df['y'], df['model_pred_BCD'])
        self.assertTrue( np.isclose(mse, 0.52, atol=0.01), f"MSE is {mse}, expected around 0.52" )
        
        # check whether the global scale converges to the mean of y
        self.assertTrue( abs( lprm_ols.global_scalar_ / df['y'].mean() -1 ) < 0.05 )

    def test_LinearProductRegressorBCD_weights( self ):
        df = self.df
        train_df = self.train_df
        feature_groups = self.feature_groups

        dcontainer = ProductModelDataContainer( df, train_df, response= df['y'], feature_groups=feature_groups )
        lprm_ols = LinearProductRegressorBCD()
        lprm_ols.fit( dcontainer, feature_groups=feature_groups, n_iterations=100, early_stopping_rounds=20, cache_qr_decomp=False, weights=np.array([1.0] * df.shape[0]) )
        df['model_pred_BCD'] = lprm_ols.predict( dcontainer )

        # check the mean squared error; whether the model converges
        mse = mean_squared_error(df['y'], df['model_pred_BCD'])
        self.assertTrue( np.isclose(mse, 0.52, atol=0.01), f"MSE is {mse}, expected around 0.52" )
        
        # check whether the global scale converges to the mean of y
        self.assertTrue( abs( lprm_ols.global_scalar_ / df['y'].mean() -1 ) < 0.05 )

class TestFrozenSubmodelBlocks(unittest.TestCase):
    """A block passed through ``submodels`` is held fixed during BCD.

    Every read path has to route through the submodel rather than the placeholder
    coefficient that init leaves behind in ``coef_``, otherwise fitting and
    prediction silently disagree about what that block contributes.
    """

    def setUp(self):
        self.df, self.toolkit = _setup_toolkit()

    def _fit_with_frozen_x1(self):
        df = self.df
        # Dropping x1 from the preprocess config is what turns its expanded block
        # into the raw column, which is the input the submodel expects.
        component_tk = self.toolkit.clone( exclude_preprocess_features=['x1'] )
        component_tk.fit( df )
        expanded_df = component_tk.get_expanded_df( df )
        self.assertEqual( component_tk.feature_groups['x1'], ['x1'] )

        grid = np.linspace( df['x1'].min(), df['x1'].max(), 40 )
        submodel = InterpolatedModel().fit( grid, 1.0 + 0.05 * np.cos( grid ) )

        dcontainer = ProductModelDataContainer(
            df, expanded_df, response=df['y'], feature_groups=component_tk.feature_groups )
        model = LinearProductRegressorBCD()
        model.fit(
            dcontainer,
            feature_groups=component_tk.feature_groups,
            submodels={'x1': submodel},
            n_iterations=20,
            early_stopping_rounds=5,
        )
        return model, dcontainer, submodel

    def test_block_mean_reads_from_submodel(self):
        model, dcontainer, submodel = self._fit_with_frozen_x1()
        block = dcontainer.get_expanded_array_for_feature_group('x1')

        self.assertAlmostEqual(
            model.block_means_['x1'], float( np.mean( submodel.predict( block ) ) ), places=10 )

    def test_frozen_block_prediction_matches_submodel(self):
        model, dcontainer, submodel = self._fit_with_frozen_x1()
        block = dcontainer.get_expanded_array_for_feature_group('x1')

        self.assertTrue( np.allclose(
            model.single_feature_group_predict('x1', dcontainer), submodel.predict( block ) ) )

    def test_full_prediction_factors_through_submodel(self):
        model, dcontainer, submodel = self._fit_with_frozen_x1()
        block = dcontainer.get_expanded_array_for_feature_group('x1')

        # Dividing the full prediction by every other block must leave exactly the
        # submodel curve times the global scalar.
        others = model.leave_out_feature_group_predict('x1', dcontainer)
        implied = model.predict( dcontainer ) / others

        self.assertTrue( np.allclose( implied, submodel.predict( block ), rtol=1e-6 ) )


class TestLinearProductRegressorScipy(unittest.TestCase):
    def setUp(self):
        df, expanded_df, feature_groups = _setup_data()
        self.df = df
        self.train_df = expanded_df
        self.feature_groups = feature_groups

    def test_LinearProductRegressorScipy(self):
        df = self.df
        train_df = self.train_df
        feature_groups = self.feature_groups

        lpm_scipy = LinearProductRegressorScipy( xtol=1e-12, gtol=1e-12, ftol=1e-12 )
        lpm_scipy.fit( X=train_df, y=df['y'], feature_groups=feature_groups, verbose=1 )
        df['model_predict_scipy'] = lpm_scipy.predict(train_df)

        # check the mean squared error
        mse = mean_squared_error(df['y'], df['model_predict_scipy'])
        self.assertTrue( np.isclose(mse, 0.52, atol=0.01), f"MSE is {mse}, expected around 0.52" )

    def test_leave_out_feature_group_predict(self):
        df = self.df
        train_df = self.train_df
        feature_groups = self.feature_groups

        lprm_ols = LinearProductRegressorScipy()
        lprm_ols.fit( train_df, df['y'], feature_groups=feature_groups )

        # leave out x1
        df['model_pred_x1_excluded'] = lprm_ols.leave_out_feature_group_predict('x1', train_df)

        # leave out x2
        df['model_pred_x2_excluded'] = lprm_ols.leave_out_feature_group_predict('x2', train_df)

        # leave out x3
        df['model_pred_x3_excluded'] = lprm_ols.leave_out_feature_group_predict('x3', train_df)

        # check the predictions and product of predictions
        df['model_pred_product'] = np.sqrt( df['model_pred_x1_excluded'] * df['model_pred_x2_excluded'] * df['model_pred_x3_excluded'] )
        df['model_pred'] = lprm_ols.predict(train_df)

        # check two predictions should be close element wise
        self.assertTrue(np.allclose(df['model_pred'], df['model_pred_product'], atol=0.01),
                        "Predictions with and without feature groups should be close.")