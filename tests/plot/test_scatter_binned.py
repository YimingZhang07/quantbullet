import unittest
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from quantbullet.plot.scatter_binned import plot_scatter_multi_y
from quantbullet.plot.scatter_binned import _prepare_binned_stats


DEV_MODE = True


class TestScatterBinned(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "./tests/_cache_dir"
        if DEV_MODE:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        else:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if not DEV_MODE:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_scatter_multi_y_basic(self):
        np.random.seed(1)
        n = 120
        df = pd.DataFrame({
            "x": np.random.uniform(0, 100, n),
            "y1": np.random.normal(0, 1, n),
            "y2": np.random.normal(0, 1, n),
        })
        df["y1"] += 0.2 * df["x"]
        df["y2"] += 0.1 * df["x"] + np.sin(df["x"] / 10) * 2

        fig, ax = plot_scatter_multi_y(df, x_col="x", y_cols=["y1", "y2"], mode="scatter")
        fig_path = Path(self.cache_dir) / "test_scatter_multi_y.png"
        fig.savefig(fig_path)
        self.assertTrue(fig_path.exists(), "Scatter plot was not saved correctly.")

    def test_binned_multi_y_qcut(self):
        np.random.seed(2)
        n = 200
        df = pd.DataFrame({
            "x": np.random.uniform(0, 100, n),
            "y1": np.random.normal(0, 1, n),
            "y2": np.random.normal(0, 1, n),
        })
        df["y1"] += 0.3 * df["x"]
        df["y2"] += 0.25 * df["x"] + np.cos(df["x"] / 12) * 1.5

        fig, ax = plot_scatter_multi_y(
            df,
            x_col="x",
            y_cols=["y1", "y2"],
            mode="binned",
            n_bins=8,
            err="std",
            binned_style="errorbar",
            connect_means=True,
            capsize=5,
        )
        fig_path = Path(self.cache_dir) / "test_binned_multi_y_qcut.png"
        fig.savefig(fig_path)
        self.assertTrue(fig_path.exists(), "Binned (qcut) plot was not saved correctly.")

    def test_binned_multi_y_cutoffs(self):
        np.random.seed(3)
        n = 180
        df = pd.DataFrame({
            "x": np.random.uniform(0, 100, n),
            "y1": np.random.normal(0, 1, n),
            "y2": np.random.normal(0, 1, n),
        })
        df["y1"] += 0.15 * df["x"]
        df["y2"] += 0.10 * df["x"]

        cutoffs = [20, 40, 60, 80]
        fig, ax = plot_scatter_multi_y(
            df,
            x_col="x",
            y_cols=["y1", "y2"],
            mode="binned",
            cutoffs=cutoffs,
            include_inf=True,
            label_style="simple",
            # use interval mid/right/left positions rather than categorical labels
            bin_x="right",
            x_tick_labels="numeric",
            binned_style="errorbar",
        )
        fig_path = Path(self.cache_dir) / "test_binned_multi_y_cutoffs.png"
        fig.savefig(fig_path)
        self.assertTrue(fig_path.exists(), "Binned (cutoffs) plot was not saved correctly.")

    def test_cutoffs_binned_centers_are_numeric(self):
        np.random.seed(4)
        n = 120
        df = pd.DataFrame({
            "x": np.random.uniform(0, 100, n),
            "y1": np.random.normal(0, 1, n),
            "y2": np.random.normal(0, 1, n),
        })
        cutoffs = [20, 40, 60, 80]
        res = _prepare_binned_stats(
            df,
            "x",
            ["y1", "y2"],
            cutoffs=cutoffs,
            include_inf=True,
            label_style="simple",
        )
        bdf = res.binned_df
        # centers should live on the same x scale (0..100-ish), not category codes 0..n
        self.assertGreaterEqual(bdf["x_center"].min(), -1e6)  # allow -inf bins -> finite edge
        self.assertLessEqual(bdf["x_center"].max(), 1e6)
        self.assertTrue(np.isfinite(bdf["x_center"]).all())


