import unittest
import shutil
from pathlib import Path
import numpy as np
from quantbullet.plot.binned_plots import plot_binned_actual_vs_pred
import pandas as pd

DEV_MODE = True

class TestPlotBinnedPlots(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "./tests/_cache_dir"
        if DEV_MODE:
            # In DEV_MODE, ensure directory exists but don't clear it
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        else:
            # In non-DEV_MODE, clear and recreate
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        # only clear cache dir in non-dev mode
        # we want to keep files for inspection in dev mode
        if not DEV_MODE:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_basic_functionality(self):
        # Create a DataFrame for testing with more data
        np.random.seed(42)
        n_rows = 100
        data = {
            'facet_col': np.random.choice(['A', 'B', 'C'], n_rows),
            'x_col': np.random.uniform(0, 100, n_rows),
            'act_col': np.random.normal(50, 15, n_rows),
            'pred_col': np.random.normal(52, 10, n_rows),
            'weight_col': np.random.randint(1, 10, n_rows) # Varying weights
        }
        df = pd.DataFrame(data)
        # Introduce some correlation for 'act' and 'pred' with 'x' to make plots look real
        df['act_col'] += df['x_col'] * 0.5
        df['pred_col'] += df['x_col'] * 0.5

        fig, axes = plot_binned_actual_vs_pred(
            df,
            facet_col='facet_col',
            x_col='x_col',
            act_col='act_col',
            pred_col='pred_col',
            weight_col='weight_col',
            n_bins=10
        )

        # Save the figure to the cache directory for inspection
        fig_path = Path(self.cache_dir) / "test_facet_scatter.png"
        fig.savefig(fig_path)

        self.assertTrue(fig_path.exists(), "Facet scatter plot was not saved correctly.")
        
    def test_no_facet_functionality(self):
        # Create a DataFrame for testing without facets
        np.random.seed(99)
        n_rows = 200
        data = {
            'x_col': np.random.uniform(0, 100, n_rows),
            'act_col': np.random.normal(50, 15, n_rows),
            'pred_col': np.random.normal(52, 10, n_rows),
            'weight_col': np.random.exponential(scale=5, size=n_rows) # More extreme weight variation
        }
        df = pd.DataFrame(data)
        df['act_col'] += np.sin(df['x_col'] / 10) * 20  # Non-linear relationship
        df['pred_col'] += np.sin(df['x_col'] / 10) * 18

        fig, axes = plot_binned_actual_vs_pred(
            df,
            # No facet_col provided
            x_col='x_col',
            act_col='act_col',
            pred_col='pred_col',
            weight_col='weight_col',
            n_bins=10
        )

        # Save the figure to the cache directory for inspection
        fig_path = Path(self.cache_dir) / "test_no_facet_scatter.png"
        fig.savefig(fig_path)

        self.assertTrue(fig_path.exists(), "No-facet scatter plot was not saved correctly.")

    def test_multi_pred_single_plot(self):
        # Multiple prediction columns, no faceting
        np.random.seed(123)
        n_rows = 150
        data = {
            'x_col': np.random.uniform(0, 100, n_rows),
            'act_col': np.random.normal(50, 12, n_rows),
            'pred_col_a': np.random.normal(50, 10, n_rows),
            'pred_col_b': np.random.normal(52, 11, n_rows),
            'weight_col': np.random.randint(1, 8, n_rows),
        }
        df = pd.DataFrame(data)
        df['act_col'] += np.cos(df['x_col'] / 12) * 15
        df['pred_col_a'] += np.cos(df['x_col'] / 12) * 13
        df['pred_col_b'] += np.cos(df['x_col'] / 12) * 11

        fig, axes = plot_binned_actual_vs_pred(
            df,
            x_col='x_col',
            act_col='act_col',
            pred_col=['pred_col_a', 'pred_col_b'],
            weight_col='weight_col',
            n_bins=8
        )

        fig_path = Path(self.cache_dir) / "test_multi_pred_single.png"
        fig.savefig(fig_path)
        self.assertTrue(fig_path.exists(), "Multi-pred single-plot was not saved correctly.")

    def test_multi_pred_with_facets(self):
        # Multiple prediction columns with faceting
        np.random.seed(456)
        n_rows = 180
        data = {
            'facet_col': np.random.choice(['A', 'B', 'C'], n_rows),
            'x_col': np.random.uniform(0, 100, n_rows),
            'act_col': np.random.normal(45, 14, n_rows),
            'pred_col_a': np.random.normal(46, 12, n_rows),
            'pred_col_b': np.random.normal(48, 10, n_rows),
            'weight_col': np.random.randint(1, 10, n_rows),
        }
        df = pd.DataFrame(data)
        df['act_col'] += df['x_col'] * 0.3
        df['pred_col_a'] += df['x_col'] * 0.28
        df['pred_col_b'] += df['x_col'] * 0.25

        fig, axes = plot_binned_actual_vs_pred(
            df,
            facet_col='facet_col',
            x_col='x_col',
            act_col='act_col',
            pred_col=['pred_col_a', 'pred_col_b'],
            weight_col='weight_col',
            n_bins=9
        )

        fig_path = Path(self.cache_dir) / "test_multi_pred_facets.png"
        fig.savefig(fig_path)
        self.assertTrue(fig_path.exists(), "Multi-pred faceted plot was not saved correctly.")