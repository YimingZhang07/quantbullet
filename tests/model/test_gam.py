import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from quantbullet.model import WrapperGAM
from quantbullet.model.gam_replay import GAMReplayModel
from quantbullet.model.feature import FeatureSpec, FeatureRole, Feature
from quantbullet.core.enums import DataType
from quantbullet.model.gam import SplineTermData, SplineByGroupTermData, TensorTermData, FactorTermData


class TestWrapperGAM(unittest.TestCase):
    """Test cases for WrapperGAM based on pygam-example.ipynb"""
    
    def setUp(self):
        """Set up test data exactly like the notebook example"""
        np.random.seed(42)
        self.n_samples = 200
        
        # Create features
        age = np.random.uniform(20, 80, self.n_samples)
        income = np.random.uniform(20000, 120000, self.n_samples)
        education = np.random.uniform(8, 20, self.n_samples)
        level = np.random.choice(['highschool', 'bachelor', 'master', 'phd'], self.n_samples)
        
        # Create target with non-linear relationships
        happiness = (
            0.5 * np.sin((age - 40) / 10) +  # non-linear relationship with age
            0.3 * np.log(income / 30000) +   # log relationship with income
            0.2 * education +                # linear relationship with education
            0.1 * (level == 'phd').astype(float) +  # categorical effect
            np.random.normal(0, 0.5, self.n_samples)  # noise
        )
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'age': age,
            'income': income,
            'education': education,
            'level': level,
            'happiness': happiness
        })
        
        # Create feature specification from the notebook
        self.features = FeatureSpec(
            features=[
                Feature(name='age', dtype=DataType.FLOAT, role=FeatureRole.MODEL_INPUT, 
                       specs={ "spline_order" : 3, "n_splines" : 6, "lam" : 0.1, "by": "level" }),
                Feature(name='income', dtype=DataType.FLOAT, role=FeatureRole.MODEL_INPUT, 
                       specs={ "spline_order" : 3, "n_splines" : 6, "lam" : 0.1, "constraints" : "monotonic_inc" } ),
                # Note: The notebook example uses education by income (float-by-float -> tensor term)
                Feature(name='education', dtype=DataType.FLOAT, role=FeatureRole.MODEL_INPUT, 
                       specs={ "spline_order" : 3, "n_splines" : 6, "lam" : 0.1, "by" : "income" } ),
                Feature(name='level', dtype=DataType.CATEGORY, role=FeatureRole.MODEL_INPUT ),
                Feature(name='happiness', dtype=DataType.FLOAT, role=FeatureRole.TARGET )
            ]
        )
    
    def test_full_pipeline(self):
        """Test fitting, prediction, partial dependence extraction, and plotting."""
        
        # 1. Fit the model
        wgam = WrapperGAM(self.features)
        wgam.fit(self.data, self.data['happiness'])
        
        # Check basic fitting
        self.assertIsNotNone(wgam.gam_)
        self.assertIn('level', wgam.category_levels_)
        
        # 2. Test Prediction
        predictions = wgam.predict(self.data)
        self.assertEqual(len(predictions), len(self.data))
        self.assertFalse(np.any(np.isnan(predictions)))
        
        # 3. Test Partial Dependence Data Extraction
        pdata = wgam.get_partial_dependence_data()
        
        # Check output structure
        # 'income' is a simple spline (constrained)
        self.assertIn('income', pdata)
        self.assertIsInstance(pdata['income'], SplineTermData)
        self.assertEqual(pdata['income'].feature, 'income')
        self.assertIsNotNone(pdata['income'].conf_lower)
        
        # 'age' by 'level' is SplineByGroupTermData
        self.assertIn(('age', 'level'), pdata)
        self.assertIsInstance(pdata[('age', 'level')], SplineByGroupTermData)
        self.assertEqual(pdata[('age', 'level')].by_feature, 'level')
        self.assertIn('bachelor', pdata[('age', 'level')].group_curves)
        
        # 'education' by 'income' is TensorTermData
        self.assertIn(('education', 'income'), pdata)
        self.assertIsInstance(pdata[('education', 'income')], TensorTermData)
        self.assertEqual(pdata[('education', 'income')].feature_x, 'education')
        self.assertEqual(pdata[('education', 'income')].feature_y, 'income')
        self.assertIsNotNone(pdata[('education', 'income')].z)
        
        # 'level' is FactorTermData
        self.assertIn('level', pdata)
        self.assertIsInstance(pdata['level'], FactorTermData)
        self.assertEqual(set(pdata['level'].categories), set(['highschool', 'bachelor', 'master', 'phd']))
        
        # 4. Test Plotting (Smoke Test)
        try:
            fig, axes = wgam.plot_partial_dependence(scale_y_axis=False, te_plot_style="contourf")
            plt.close(fig)
        except Exception as e:
            self.fail(f"plot_partial_dependence raised exception: {e}")

    def test_replay_model_accuracy(self):
        """Test that GAMReplayModel reproduces WrapperGAM predictions closely."""
        # Fit model
        wgam = WrapperGAM(self.features)
        wgam.fit(self.data, self.data['happiness'])
        
        # Get predictions
        original_preds = wgam.predict(self.data)
        
        # Create Replay Model
        pdata = wgam.get_partial_dependence_data()
        replay = GAMReplayModel(pdata, intercept=wgam.intercept_)
        replay_preds = replay.predict(self.data)
        
        # Check accuracy
        # We use a relatively loose tolerance because:
        # 1. pygam uses B-splines directly
        # 2. We export partial dependence on a discrete grid (200 points)
        # 3. We interpolate that grid using PCHIP
        # This double approximation introduces error, but it should be small.
        # rtol=1e-2 (1%) is reasonable for this approximation.
        np.testing.assert_allclose(original_preds, replay_preds, rtol=0.01, atol=0.05)
        
        # Test Out-of-Sample Extrapolation Safety (Flat extrapolation)
        # Create data points far outside the training range
        oob_data = self.data.iloc[:5].copy()
        oob_data['income'] = 1_000_000.0  # Way higher than training max ~120k
        oob_data['age'] = 150.0           # Way higher than training max ~80
        
        # Predict
        oob_preds = replay.predict(oob_data)
        self.assertEqual(len(oob_preds), 5)
        self.assertFalse(np.any(np.isnan(oob_preds)))
        self.assertFalse(np.any(np.isinf(oob_preds)))

if __name__ == '__main__':
    unittest.main()
