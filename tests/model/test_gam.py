import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from quantbullet.model import WrapperGAM
from quantbullet.model.feature import FeatureSpec, FeatureRole, Feature
from quantbullet.core.enums import DataType


class TestWrapperGAM(unittest.TestCase):
    """Test cases for WrapperGAM based on pygam-example.ipynb"""
    
    def setUp(self):
        """Set up test data similar to the notebook example"""
        np.random.seed(42)
        self.n_samples = 200
        
        # Create features similar to the notebook
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
        
        # Create feature specification similar to the notebook
        self.features = FeatureSpec(
            features=[
                Feature(
                    name='age', 
                    dtype=DataType.FLOAT, 
                    role=FeatureRole.MODEL_INPUT, 
                    specs={"spline_order": 3, "n_splines": 6, "lam": 0.1, "by": "level"}
                ),
                Feature(
                    name='income', 
                    dtype=DataType.FLOAT, 
                    role=FeatureRole.MODEL_INPUT, 
                    specs={"spline_order": 3, "n_splines": 6, "lam": 0.1}
                ),
                Feature(
                    name='education', 
                    dtype=DataType.FLOAT, 
                    role=FeatureRole.MODEL_INPUT, 
                    specs={"spline_order": 3, "n_splines": 6, "lam": 0.1}
                ),
                Feature(
                    name='level', 
                    dtype=DataType.CATEGORY, 
                    role=FeatureRole.SECONDARY_INPUT
                ),
                Feature(
                    name='happiness', 
                    dtype=DataType.FLOAT, 
                    role=FeatureRole.TARGET
                )
            ]
        )
    
    def test_fit_basic(self):
        """Test basic fitting functionality"""
        wgam = WrapperGAM( self.features )
        
        # Fit the model
        result = wgam.fit( self.data, self.data[ 'happiness' ] )
        
        # Check that category levels are stored for categorical features
        self.assertIn( 'level', wgam.category_levels_ )
        self.assertEqual( len( wgam.category_levels_[ 'level' ] ), 4 )  # 4 education levels
    
    def test_predict(self):
        """Test prediction functionality"""
        wgam = WrapperGAM( self.features )
        wgam.fit( self.data, self.data[ 'happiness' ] )
        
        # Make predictions
        predictions = wgam.predict( self.data )
        
        # Check predictions shape and type
        self.assertEqual( len( predictions ), len( self.data ) )
        self.assertIsInstance( predictions, np.ndarray )
        
        # Check that predictions are reasonable (not NaN or infinite)
        self.assertFalse( np.any( np.isnan( predictions ) ) )
        self.assertFalse( np.any( np.isinf( predictions ) ) )