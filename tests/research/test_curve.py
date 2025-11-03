# fmt: off
import unittest
import shutil
import pandas as pd
from pathlib import Path
from quantbullet.research.curve import MVOCCurve

class TestCurveModel( unittest.TestCase ):
    def setUp( self ):
        self.cache_dir = "./tests/_cache_dir"
        # just remove all files in the cache dir, but not the dir itself
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        pass
    
    def test_build_curve( self ):
        left_bound      = 0
        right_bound     = 1000
        step_size       = 100
        x_name          = "MVOC"
        y_name          = "RiskSpread"

        curve_model = MVOCCurve( left_bound, right_bound, step_size, x_name, y_name )
        
        x_values = [ 50, 150, 250, 350, 450, 550, 650, 750, 850, 950 ]
        y_values = [ 200, 180, 160, 150, 140, 130, 125, 120, 115, 110 ]
        dates = [ pd.to_datetime("2025-09-30") ] * len( x_values )
        
        curve_df = curve_model.build_curve( x_values, y_values, dates )
        
        self.assertIsNotNone( curve_df )
        self.assertEqual( curve_df.shape[0], len( x_values ) )
        self.assertIn( 'x', curve_df.columns )
        self.assertIn( 'y', curve_df.columns )
        self.assertIn( 'count', curve_df.columns )

        # now plot the curve and save to cache dir
        fig, ax = curve_model.plot_curve()
        fig_path = Path(self.cache_dir) / "mvoc_risk_spread_curve.png"
        fig.savefig( fig_path )
        self.assertTrue( fig_path.exists() )