import unittest
import shutil
from pathlib import Path
from quantbullet.plot.model_perf import plot_facet_scatter
import pandas as pd

DEV_MODE = True

class TestPlotModelPerf(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "./tests/_cache_dir"
        # just remove all files in the cache dir, but not the dir itself
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        pass
    
    def tearDown(self):
        # only clear cache dir in non-dev mode
        # we want to keep files for inspection in dev mode
        if not DEV_MODE:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def test_basic_functionality(self):
        # Create a simple DataFrame for testing
        data = {
            'facet_col': ['A', 'A', 'B', 'B', 'A', 'B'],
            'x_col': [1, 2, 1, 2, 3, 3],
            'act_col': [10, 20, 15, 25, 30, 35],
            'pred_col': [12, 18, 14, 26, 28, 36],
            'weight_col': [1, 1, 1, 1, 1, 1]
        }
        df = pd.DataFrame(data)

        fig, axes = plot_facet_scatter(
            df,
            facet_colname='facet_col',
            x_colname='x_col',
            act_colname='act_col',
            pred_colname='pred_col',
            weight_colname='weight_col',
            n_bins=2
        )

        # Save the figure to the cache directory for inspection
        fig_path = Path(self.cache_dir) / "test_facet_scatter.png"
        fig.savefig(fig_path)

        self.assertTrue(fig_path.exists(), "Facet scatter plot was not saved correctly.")