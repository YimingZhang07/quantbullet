"""Numerical contracts plus a reproducible visual gallery.

Run: python -m pytest tests/plot/test_grouped_means.py -q
Images and gallery.html are saved under tests/_cache_dir/grouped_means/.
Synthetic rates are proportions; incentive is in percentage points.
"""
from dataclasses import replace
from html import escape
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from quantbullet.plot import (
    BinSpec, MINIMAL_THEME, draw_grouped_means,
    plot_grouped_means, summarize_grouped_means,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "_cache_dir" / "grouped_means"


def make_fake_mortgage_data(n=12000, seed=731):
    """Seeded demonstration data, not a fitted or financial forecast model."""
    rng = np.random.default_rng(seed)
    vintage = rng.choice([2019, 2020, 2021], size=n, p=[0.25, 0.5, 0.25])
    channel = rng.choice(["Retail", "Broker"], size=n, p=[0.7, 0.3])
    occupancy = rng.choice(["Owner", "Investor"], size=n, p=[0.8, 0.2])
    incentive = np.clip(rng.normal(0.7 + 0.25 * (vintage - 2020), 0.8, n), -1, 3)
    fico = np.clip(rng.normal(735, 40, n), 600, 850)
    upb = rng.lognormal(np.log(250000), 0.5, n)
    base = 0.035 + 0.22 / (1 + np.exp(-2.3 * (incentive - 0.7)))
    adjustment = 0.014 * (vintage - 2020) + 0.012 * (channel == "Broker") - 0.023 * (occupancy == "Investor") + 0.0002 * (fico - 735)
    historical = np.clip(base + adjustment + rng.normal(0, 0.025, n), 0, 0.6)
    model = np.clip(base + adjustment * 0.8 + 0.008 * np.tanh(incentive - 1), 0, 0.6)
    return pd.DataFrame({
        "incentive": incentive, "historical_cpr": historical, "model_cpr": model,
        "upb": upb, "vintage": vintage, "channel": channel,
        "occupancy": occupancy, "fico": fico,
    })


class TestGroupedMeansStatistics(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_weighted_means_use_metric_specific_valid_rows(self):
        df = pd.DataFrame({"x": [0] * 5, "a": [1, 3, np.nan, 5, 7],
                           "b": [2, np.nan, 4, 8, 10], "w": [1, 3, 2, np.nan, 0]})
        before = df.copy(deep=True)
        data = summarize_grouped_means(df, x="x", y=["a", "b"], weight="w")
        row = data.summary.iloc[0]
        self.assertEqual(row["count"], 5)
        self.assertAlmostEqual(row["a__mean"], 2.5)
        self.assertAlmostEqual(row["b__mean"], 10 / 3)
        self.assertEqual(row["a__valid_count"], 3)  # includes the zero-weight row
        self.assertEqual(row["a__weight_sum"], 4)
        self.assertEqual(row["b__weight_sum"], 3)
        assert_frame_equal(df, before)

    def test_unweighted_and_zero_weight(self):
        df = pd.DataFrame({"x": [0, 0], "y": [1., 3.], "w": [0., 0.]})
        self.assertEqual(summarize_grouped_means(df, x="x", y="y").summary.iloc[0]["y__mean"], 2)
        weighted = summarize_grouped_means(df, x="x", y="y", weight="w").summary.iloc[0]
        self.assertTrue(np.isnan(weighted["y__mean"]))
        self.assertEqual(weighted["y__weight_sum"], 0)

    def test_null_and_nonfinite_metrics_and_missing_group_keys(self):
        df = pd.DataFrame({"x": [0., 0., 0., np.nan], "y": [1., np.inf, np.nan, 5.],
                           "null": [None] * 4, "w": [1., 2., np.inf, 1.]})
        data = summarize_grouped_means(df, x="x", y=["y", "null"], weight="w")
        row = data.summary.iloc[0]
        self.assertEqual(row["count"], 3)
        self.assertEqual(row["y__mean"], 1)
        self.assertEqual(row["y__valid_count"], 1)
        self.assertTrue(np.isnan(row["null__mean"]))
        self.assertEqual(row["null__valid_count"], 0)
        self.assertEqual(data.excluded_count, 1)

    def test_edges_include_lowest_and_keep_empty_bin(self):
        df = pd.DataFrame({"x": [-1., 0., 1., 3., 5., np.nan], "y": np.arange(6.)})
        data = summarize_grouped_means(df, x="x", y="y", bins={"x": BinSpec.edges(np.array([0, 1, 2, 4]))})
        self.assertEqual(data.summary["count"].tolist(), [2, 0, 1])
        self.assertTrue(np.isnan(data.summary.iloc[1]["y__mean"]))
        self.assertEqual(data.excluded_count, 3)
        self.assertIn(0, data.levels["x"][0])
        np.testing.assert_array_equal(data.x_positions, [0.5, 1.5, 3.])

    def test_quantiles_are_global_not_per_facet(self):
        df = pd.DataFrame({"x": [0., 1., 2., 3., 10., 11., 12., 13.], "g": [0] * 4 + [1] * 4, "y": range(8)})
        data = summarize_grouped_means(df, x="x", y="y", col="g", bins={"x": BinSpec.quantile(2)})
        self.assertEqual(data.bin_info["x"]["edges"], (0., 6.5, 13.))
        self.assertEqual(data.summary.loc[data.summary["col"] == 0, "count"].tolist(), [4, 0])
        self.assertEqual(data.summary.loc[data.summary["col"] == 1, "count"].tolist(), [0, 4])

    def test_step_and_constant_quantile(self):
        df = pd.DataFrame({"x": [-0.7, 0., 0.7], "y": [1., 2., 3.]})
        data = summarize_grouped_means(df, x="x", y="y", bins={"x": BinSpec.step(0.5)})
        np.testing.assert_allclose(data.bin_info["x"]["edges"], [-1, -0.5, 0, 0.5, 1])
        self.assertEqual(data.summary["count"].tolist(), [1, 1, 0, 1])
        df["x"] = 7
        constant = summarize_grouped_means(df, x="x", y="y", bins={"x": BinSpec.quantile(10)})
        self.assertEqual(constant.summary["count"].tolist(), [3])
        self.assertEqual(constant.summary["y__mean"].tolist(), [2])

    def test_pandas_polars_parity_for_each_binning_strategy(self):
        try:
            import polars as pl
        except ImportError:
            self.skipTest("Polars not installed")
        records = {"x": [0., 0.2, 0.8, 1., 1.5, None], "g": [0, 0, 1, 1, 1, 1],
                   "y": [1., None, 3., 4., 5., 6.], "z": [2., 4., None, 8., 10., 12.],
                   "w": [1., 2., 0., None, 4., 1.]}
        for spec in [None, BinSpec.edges([0, 0.5, 1, 2]), BinSpec.quantile(3), BinSpec.step(0.5)]:
            with self.subTest(spec=spec):
                kwargs = dict(x="x", y=["y", "z"], weight="w", group="g", bins={"x": spec} if spec else None)
                pandas_data = summarize_grouped_means(pd.DataFrame(records), **kwargs)
                polars_data = summarize_grouped_means(pl.DataFrame(records), **kwargs)
                assert_frame_equal(pandas_data.summary, polars_data.summary)
                self.assertEqual(pandas_data.bin_info, polars_data.bin_info)
                self.assertEqual(pandas_data.excluded_count, polars_data.excluded_count)

    def test_rejects_invalid_inputs(self):
        df = pd.DataFrame({"x": [0], "y": [1.], "w": [-1.], "a": [0], "b": [0], "c": [0]})
        for kwargs in [{"weight": "w"}, {"y": []}, {"y": "missing"},
                       {"group": "a", "row": "b", "col": "c"}, {"group": "x"},
                       {"bins": {"y": BinSpec.quantile(2)}}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                summarize_grouped_means(df, **(dict(x="x", y="y") | kwargs))
        for factory in [lambda: BinSpec.step(0), lambda: BinSpec.quantile(0),
                        lambda: BinSpec.edges([1, 1]), lambda: BinSpec.edges([0, np.inf])]:
            with self.assertRaises(ValueError):
                factory()

    def test_empty_and_all_missing_inputs_render(self):
        for df in [pd.DataFrame({"x": [], "y": [], "g": []}),
                   pd.DataFrame({"x": [np.nan], "y": [1.], "g": [0]})]:
            for facet in [{}, {"col": "g"}, {"row": "g"}]:
                with self.subTest(facet=facet):
                    result = plot_grouped_means(df, x="x", y="y", **facet)
                    result.fig.canvas.draw()
                    self.assertEqual(result.summary["count"].sum(), 0)
                    self.assertEqual(result.axes.shape, (1, 1))

    def test_falsey_facets_do_not_mix_and_count_not_duplicated(self):
        df = pd.DataFrame({"x": [1, 2, 1, 2], "a": [10., 20., 30., 40.],
                           "b": [11., 21., 31., 41.], "g": [0, 0, 1, 1]})
        result = plot_grouped_means(df, x="x", y=["a", "b"], col="g")
        np.testing.assert_array_equal(result.axes[0, 0].lines[0].get_ydata(), [10, 20])
        np.testing.assert_array_equal(result.axes[0, 1].lines[0].get_ydata(), [30, 40])
        self.assertEqual(sum(p.get_height() for ax in result.count_axes.flat for p in ax.patches), 4)
        self.assertIn("g: 0", result.axes[0, 0].get_title())

    def test_stacked_counts_and_stable_group_colors(self):
        df = pd.DataFrame({"x": [1, 1, 2, 2], "y": [1., 2., 3., 4.], "g": ["A", "B", "A", "B"], "c": [0, 0, 1, 1]})
        result = plot_grouped_means(df, x="x", y="y", group="g", col="c", count_mode="stacked", share_count_y=True)
        self.assertEqual(sum(p.get_height() for ax in result.count_axes.flat for p in ax.patches), len(df))
        for left, right in zip(result.axes[0, 0].lines, result.axes[0, 1].lines):
            self.assertEqual(left.get_color(), right.get_color())
        self.assertEqual(result.count_axes[0, 0].get_ylim(), result.count_axes[0, 1].get_ylim())

    def test_ordered_numeric_categories_and_theme(self):
        df = pd.DataFrame({"x": pd.Categorical([20, 10], categories=[20, 10, 30], ordered=True), "y": [1., 2.]})
        theme = replace(MINIMAL_THEME, title_fontsize=17, label_fontsize=14, legend_fontsize=11)
        result = plot_grouped_means(df, x="x", y="y", theme=theme, count_mode="none")
        np.testing.assert_array_equal(result.data.x_positions, [0, 1, 2])
        self.assertEqual([t.get_text() for t in result.axes[0, 0].get_xticklabels()], ["20", "10", "30"])
        self.assertEqual(result.axes[0, 0].xaxis.label.get_fontsize(), 14)
        self.assertEqual(result.fig.legends[0].get_texts()[0].get_fontsize(), 11)
        self.assertIsNone(result.count_axes[0, 0])

    def test_missing_matrix_combination_and_reusable_aggregation(self):
        df = pd.DataFrame({"x": [1., 2.], "y": [1., 2.], "r": [0, 1], "c": [0, 1]})
        data = summarize_grouped_means(df, x="x", y="y", row="r", col="c")
        result = draw_grouped_means(data)
        self.assertIs(result.data, data)
        self.assertEqual(result.axes.shape, (2, 2))
        self.assertIn("No observations", [text.get_text() for text in result.axes[0, 1].texts])
        self.assertEqual(sum(p.get_height() for ax in result.count_axes.flat for p in ax.patches), 2)


class TestGroupedMeansGallery(unittest.TestCase):
    """One reproducible scenario per unittest, with saved figures to inspect."""

    @classmethod
    def setUpClass(cls):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.df = make_fake_mortgage_data()
        cls.cards = []

    @classmethod
    def tearDownClass(cls):
        cards = "\n".join(
            f'<section><h2>{escape(title)}</h2><p>{escape(note)}</p><a href="{name}.png"><img src="{name}.png" alt="{escape(title)}"></a></section>'
            for name, title, note in sorted(cls.cards)
        )
        html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Grouped means gallery</title>
<style>body{font:16px/1.6 system-ui;background:#f3f5f7;color:#1a2635;max-width:1400px;margin:40px auto;padding:0 24px}h1{font-size:32px}section{background:white;padding:24px;margin:28px 0;border-radius:12px}h2{margin:0}p{color:#596575}img{max-width:100%;display:block;margin:auto}code{background:#e7edf4;padding:3px 7px}</style>
<h1>Weighted means + sample counts</h1><p>12,000 synthetic mortgage records · fixed seed 731 · UPB weighted CPR.<br>Lines use the left axis; background bars show row counts on the right axis. Data is synthetic, not a forecast.</p>
''' + cards + "</html>"
        (OUTPUT_DIR / "gallery.html").write_text(html, encoding="utf-8")

    def tearDown(self):
        plt.close("all")

    def save_case(self, name, title, note, **kwargs):
        defaults = dict(x="incentive", y=["historical_cpr", "model_cpr"], weight="upb",
                        bins={"incentive": BinSpec.step(0.25)}, y_format=".0%", ylabel="CPR (UPB weighted)",
                        title=title, labels={"incentive": "Refinance incentive (pp)", "historical_cpr": "Historical CPR", "model_cpr": "Model CPR", "vintage": "Vintage", "channel": "Channel", "occupancy": "Occupancy", "fico": "FICO"})
        result = plot_grouped_means(self.df, **(defaults | kwargs))
        path = OUTPUT_DIR / f"{name}.png"
        result.fig.savefig(path, dpi=130)
        self.assertTrue(path.exists())
        self.assertEqual(result.summary["count"].sum(), len(self.df))
        self.cards.append((name, title, note))
        return result

    def test_01_basic_comparison(self):
        self.save_case("01_basic", "Historical vs model CPR", "One panel, two weighted means, total count overlay.", panel_size=(8, 4.8))

    def test_02_overlapped_vintages(self):
        self.save_case("02_overlap", "Vintage comparison on one chart", "Color identifies vintage; solid is historical and dashed is model. Bars count all vintages together.", group="vintage", panel_size=(9, 5.2))

    def test_03_stacked_count_overlay(self):
        self.save_case("03_stacked", "Vintage curves with count composition", "Count bars are stacked by vintage; totals are not duplicated across the two CPR metrics.", group="vintage", count_mode="stacked", panel_size=(9, 5.2))

    def test_04_wrapped_facets(self):
        result = self.save_case("04_facets", "One panel per vintage", "Identical incentive bins and shared CPR scales. The unused wrapped slot is hidden.", col="vintage", wrap=2)
        self.assertEqual(result.axes.shape, (2, 2))
        self.assertFalse(result.axes[1, 1].get_visible())

    def test_05_matrix(self):
        result = self.save_case("05_matrix", "Channel x occupancy", "Rows identify channel; columns identify occupancy. CPR and count scales are shared across panels.", row="channel", col="occupancy", share_count_y=True)
        self.assertEqual(result.axes.shape, (2, 2))

    def test_06_group_and_binned_facet(self):
        self.save_case("06_group_facet", "Vintage comparison within FICO bands", "A binned facet dimension plus overlapped vintage curves, with global quantile bins on incentive.", group="vintage", col="fico", wrap=2,
                       bins={"incentive": BinSpec.quantile(12), "fico": BinSpec.edges([600, 700, 740, 780, 850])})

    def test_07_single_metric_categorical(self):
        self.save_case("07_categorical", "Historical CPR by channel", "A single metric with an unbinned categorical x axis and one panel per occupancy.", x="channel", y="historical_cpr", col="occupancy", bins={})


if __name__ == "__main__":
    unittest.main()
