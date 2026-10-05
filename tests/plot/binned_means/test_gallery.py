"""Gallery of directly copyable plot_binned_means examples.

Run: python -m pytest tests/plot/binned_means/test_gallery.py -q
Set QB_TEST_KEEP_ARTIFACTS=1 to keep images and gallery.html under
tests/_cache_dir/binned_means/. Otherwise they are temporary.
Synthetic rates are proportions; incentive is in percentage points.
"""
import ast
from html import escape
import inspect
from textwrap import dedent
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantbullet.plot import BinSpec, plot_binned_means
from tests.artifacts import gallery_dir


# Gallery index sections, in display order; case names sort within this order.
GALLERY_SECTIONS = {
    "single": "单图 · Single panel",
    "facets": "一维分面 · Facets",
    "matrix": "矩阵分面 · Row x col matrix",
    "categorical": "分类 x 轴 · Categorical x",
}
# Call arguments that only label or size a figure; tags list the rest.
_STYLE_ARGUMENTS = {"weight", "labels", "title", "ylabel", "y_format", "panel_size"}


def _plot_call(test_method):
    """Return a test's source and the plot_binned_means call inside it."""
    source = dedent(inspect.getsource(test_method))
    call = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "plot_binned_means"
    )
    return source, call


def _plot_call_source(test_method):
    """Display the exact plot_binned_means call beside its generated image."""
    source, call = _plot_call(test_method)
    return "result = " + ast.get_source_segment(source, call)


def _plot_call_tags(test_method):
    """Short parameter tags, read from the call so they cannot drift from it."""
    tags = []
    for keyword in _plot_call(test_method)[1].keywords:
        name, value = keyword.arg, keyword.value
        if name in _STYLE_ARGUMENTS:
            continue
        if name == "y":
            count = len(value.elts) if isinstance(value, (ast.List, ast.Tuple)) else 1
            tags.append(f"y: {count} metric{'s' if count > 1 else ''}")
        elif name == "bins":
            specs = [f"{key.value} {spec.func.attr}" for key, spec in zip(value.keys, value.values)]
            tags.append("bins: " + (", ".join(specs) or "none"))
        elif isinstance(value, ast.Constant):
            tags.append(name if value.value is True else f"{name}={value.value}")
        else:
            tags.append(name)
    return tags


def make_fake_mortgage_data(n=40000, seed=731):
    """Seeded demonstration data, not a fitted or financial forecast model."""
    rng = np.random.default_rng(seed)
    vintage = rng.choice([2019, 2020, 2021], size=n, p=[0.25, 0.5, 0.25])
    channel = rng.choice(["Retail", "Broker"], size=n, p=[0.7, 0.3])
    occupancy = rng.choice(["Owner", "Investor"], size=n, p=[0.8, 0.2])
    incentive = np.clip(rng.normal(0.7 + 0.25 * (vintage - 2020), 0.8, n), -1, 3)
    fico = np.clip(rng.normal(735, 40, n), 600, 850)
    upb = rng.lognormal(np.log(250000), 0.5, n)
    noise = rng.normal(0, 0.025, n)
    # Drawn after the original inputs so they keep their seeded values.
    purposes = ["Purchase", "Rate/Term Refi", "Cash-out Refi"]
    purpose = rng.choice(purposes, size=n, p=[0.55, 0.25, 0.2])
    # Rate/term refis respond most to incentive; cash-outs least, but turn over faster out of the money.
    sensitivity = np.select([purpose == "Rate/Term Refi", purpose == "Cash-out Refi"], [1.25, 0.7], 1.0)
    base = 0.035 + 0.22 * sensitivity / (1 + np.exp(-2.3 * (incentive - 0.7)))
    adjustment = (0.014 * (vintage - 2020) + 0.012 * (channel == "Broker") - 0.023 * (occupancy == "Investor")
                  + 0.0002 * (fico - 735) + 0.02 * (purpose == "Cash-out Refi"))
    historical = np.clip(base + adjustment + noise, 0, 0.6)
    model = np.clip(base + adjustment * 0.8 + 0.008 * np.tanh(incentive - 1), 0, 0.6)
    return pd.DataFrame({
        "incentive": incentive, "historical_cpr": historical, "model_cpr": model,
        "historical_smm": 1 - (1 - historical) ** (1 / 12), "upb": upb, "vintage": vintage, "channel": channel,
        "occupancy": occupancy, "fico": fico,
        "purpose": pd.Categorical(purpose, categories=purposes, ordered=True),
    })


class TestBinnedMeansGallery(unittest.TestCase):
    """One reproducible scenario per unittest, with saved figures to inspect."""

    @classmethod
    def setUpClass(cls):
        cls.output_dir = gallery_dir(cls)
        cls.df = make_fake_mortgage_data()
        cls.cards = []

    @classmethod
    def tearDownClass(cls):
        cards = sorted(cls.cards)
        tags = {name: _plot_call_tags(getattr(cls, method)) for name, *_, method in cards}
        index = "\n".join(
            f"<div><h3>{escape(label)}</h3>" + "".join(
                f'<a href="#{escape(name, quote=True)}">{name[:2]} · {escape(zh)} · {escape(en)}'
                f'<small>{escape(" · ".join(tags[name]))}</small></a>'
                for name, card_section, (zh, en), _, _ in cards if card_section == section
            ) + "</div>"
            for section, label in GALLERY_SECTIONS.items()
        )
        sections = "\n".join(
            f'<section id="{escape(name, quote=True)}"><div class="eyebrow">{escape(GALLERY_SECTIONS[section])}</div>'
            f"<h2>{name[:2]} · {escape(zh)} <span>{escape(en)}</span></h2>"
            '<div class="tags">' + "".join(f"<code>{escape(tag)}</code>" for tag in tags[name]) + "</div>"
            f'<p>{escape(note)}</p><a href="{name}.png"><img src="{name}.png" alt="{escape(zh)} · {escape(en)}"></a>'
            f"<pre><code>{escape(_plot_call_source(getattr(cls, method)))}</code></pre></section>"
            for name, section, (zh, en), note, method in cards
        )
        html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Binned means gallery</title>
<style>body{font:16px/1.6 system-ui;background:#f3f5f7;color:#1a2635;max-width:1400px;margin:40px auto;padding:0 24px}h1{font-size:32px}section{background:white;padding:24px;margin:28px 0;border-radius:12px}h2{margin:0}h2 span{color:#596575;font-weight:normal}p{color:#596575}.eyebrow{font-size:13px;color:#7a8696}nav{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px 24px;padding:16px 0}nav h3{margin:0 0 4px;font-size:15px;color:#596575}nav a{display:block;color:#2e45b8;text-decoration:none;padding:4px 0}nav small{display:block;color:#7a8696;font-size:12px;line-height:1.4}.tags{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 0}.tags code{background:#e7edf4;border-radius:6px;padding:1px 8px;font:12px/1.7 ui-monospace,Consolas,monospace}img{max-width:100%;display:block;margin:auto}pre{background:#e7edf4;padding:16px;overflow-x:auto;border-radius:8px;font:13px/1.5 ui-monospace,Consolas,monospace}</style>
<h1>Binned means + sample counts</h1><p>''' + f"{len(cls.df):,}" + ''' synthetic mortgage records · fixed seed 731 · UPB weighted CPR.<br>Lines and points use the left axis; background bars show row counts on the right axis. Data is synthetic, not a forecast.<br>Cases are grouped by layout; tags list the parameters that shape each figure. The code below is extracted from each test case; replace <code>self.df</code> with your own DataFrame.</p>
<nav aria-label="Chart examples">''' + index + "</nav>" + sections + "</html>"
        (cls.output_dir / "gallery.html").write_text(html, encoding="utf-8")

    def tearDown(self):
        plt.close("all")

    def save_case(self, name, section, title, note, result):
        """Save one card; ``title`` is a (Chinese, English) pair."""
        self.assertIn(section, GALLERY_SECTIONS)
        path = self.output_dir / f"{name}.png"
        result.fig.savefig(path, dpi=130)
        self.assertTrue(path.exists())
        self.assertEqual(result.summary["count"].sum(), len(self.df))
        self.cards.append((name, section, title, note, self._testMethodName))

    def test_01_single_panel_metrics(self):
        result = plot_binned_means(
            self.df,
            x="incentive",
            y=["historical_cpr", "model_cpr"],
            weight="upb",
            bins={"incentive": BinSpec.step(0.25)},
            bar_mode="total",
            panel_size=(8, 4.8),
            y_format=".0%",
            y_label="CPR (UPB weighted)",
            title="Historical vs model CPR",
            labels={"incentive": "Refinance incentive (pp)",
                    "historical_cpr": "Historical CPR", "model_cpr": "Model CPR"},
        )
        self.save_case("01_single_metrics", "single", ("多指标对比", "Multiple metrics"),
                       "y=[...] draws multiple weighted means; bar_mode='total' overlays one count series on the right axis.", result)

    def test_02_single_panel_groups(self):
        result = plot_binned_means(
            self.df,
            x="incentive",
            y=["historical_cpr", "model_cpr"],
            weight="upb",
            bins={"incentive": BinSpec.step(0.25)},
            group="vintage",
            bar_mode="total",
            panel_size=(9, 5.2),
            y_format=".0%",
            y_label="CPR (UPB weighted)",
            title="Vintage comparison on one chart",
            labels={"incentive": "Refinance incentive (pp)", "vintage": "Vintage",
                    "historical_cpr": "Historical CPR", "model_cpr": "Model CPR"},
        )
        self.save_case("02_single_groups", "single", ("分组叠线", "Groups overlaid"),
                       "group=... overlays group curves in one panel; color identifies groups and line style identifies metrics.", result)

    def test_03_single_panel_groups_stacked(self):
        result = plot_binned_means(
            self.df,
            x="incentive",
            y=["historical_cpr", "model_cpr"],
            weight="upb",
            bins={"incentive": BinSpec.step(0.25)},
            group="vintage",
            bar_mode="stacked",
            panel_size=(9, 5.2),
            y_format=".0%",
            y_label="CPR (UPB weighted)",
            title="Vintage curves with count composition",
            labels={"incentive": "Refinance incentive (pp)", "vintage": "Vintage",
                    "historical_cpr": "Historical CPR", "model_cpr": "Model CPR"},
        )
        self.save_case("03_single_groups_stacked", "single", ("分组叠线 + 堆叠计数", "Groups + stacked counts"),
                       "Case 02 with bar_mode='stacked': the count bars split by group, in the group colors, "
                       "so each vintage's share of every bin is visible.", result)

    def test_04_single_panel_groups_stacked_weight(self):
        result = plot_binned_means(
            self.df,
            x="incentive",
            y=["historical_cpr", "model_cpr"],
            weight="upb",
            bins={"incentive": BinSpec.step(0.25)},
            group="vintage",
            bar_mode="stacked",
            bar_value="weight",
            panel_size=(9, 5.2),
            y_format=".0%",
            y_label="CPR (UPB weighted)",
            title="Vintage curves with UPB composition",
            labels={"incentive": "Refinance incentive (pp)", "vintage": "Vintage", "upb": "UPB",
                    "historical_cpr": "Historical CPR", "model_cpr": "Model CPR"},
        )
        self.save_case("04_single_groups_stacked_weight", "single", ("分组叠线 + 堆叠 UPB", "Groups + stacked UPB"),
                       "Case 03 with bar_value='weight': bars add up the weight column (UPB) instead of counting "
                       "rows, so larger loans count for more. The right-axis title and legend take the weight's "
                       "label from labels; min_count would still count rows.", result)
        self.assertEqual(result.bar_axes[0, 0].get_ylabel(), "UPB")
        np.testing.assert_allclose(sum(bar.get_height() for bar in result.bar_axes[0, 0].patches),
                                   self.df["upb"].sum(), rtol=1e-12)

    def test_05_facets_metrics(self):
        result = plot_binned_means(
            self.df,
            x="incentive",
            y=["historical_cpr", "model_cpr"],
            weight="upb",
            bins={"incentive": BinSpec.step(0.25)},
            col="vintage",
            wrap=2,
            bar_mode="total",
            y_format=".0%",
            y_label="CPR (UPB weighted)",
            title="One panel per vintage",
            labels={"incentive": "Refinance incentive (pp)", "vintage": "Vintage",
                    "historical_cpr": "Historical CPR", "model_cpr": "Model CPR"},
        )
        self.save_case("05_facets_metrics", "facets", ("换行分面 · 多指标", "Wrapped facets · multiple metrics"),
                       "col=... creates one panel per category; wrap=2 arranges panels in two columns.", result)
        self.assertEqual(result.axes.shape, (2, 2))
        self.assertFalse(result.axes[1, 1].get_visible())

    def test_06_facets_groups_binned(self):
        result = plot_binned_means(
            self.df,
            x="incentive",
            y=["historical_cpr", "model_cpr"],
            weight="upb",
            bins={"incentive": BinSpec.quantile(12),
                  "fico": BinSpec.edges([600, 700, 740, 780, 850])},
            group="vintage",
            col="fico",
            wrap=2,
            bar_mode="total",
            y_format=".0%",
            y_label="CPR (UPB weighted)",
            title="Vintage comparison within FICO bands",
            labels={"incentive": "Refinance incentive (pp)", "vintage": "Vintage",
                    "fico": "FICO", "historical_cpr": "Historical CPR",
                    "model_cpr": "Model CPR"},
        )
        self.save_case("06_facets_groups_binned", "facets", ("分组叠线 · 分位数与区间分箱", "Groups · quantile and edge bins"),
                       "BinSpec.quantile(12) gives x bins with equal row counts overall, hence the uneven bar widths; "
                       "BinSpec.edges([...]) cuts the col=fico dimension into four panels; group=... overlays vintages in each.", result)

    def test_07_facets_groups_smm(self):
        result = plot_binned_means(
            self.df,
            x="incentive",
            y="historical_smm",
            weight="upb",
            bins={"incentive": BinSpec.step(0.25)},
            group="purpose",
            col="vintage",
            bar_mode="stacked",
            bar_scale="shared",
            min_count=30,
            y_transform=lambda smm: 1 - (1 - smm) ** 12,
            y_format=".0%",
            y_label="CPR (UPB weighted)",
            title="CPR by loan purpose within each vintage",
            labels={"incentive": "Refinance incentive (pp)", "purpose": "Purpose",
                    "vintage": "Vintage", "historical_smm": "CPR"},
            y_ticks="outer",
            bar_ticks="outer",
            y_titles="outer",
            bar_titles="outer",
        )
        self.save_case("07_facets_groups_smm", "facets", ("分组叠线 · 单指标 SMM→CPR", "Groups · one metric, SMM → CPR"),
                       "One metric with group=... overlays loan purposes inside each col=... vintage panel. "
                       "y_transform converts each bin's weighted SMM to CPR (weight first, then convert); "
                       "purpose is an ordered pandas Categorical, so the legend follows its declared order; "
                       "min_count=30 breaks curves at sparse bins but keeps their bars; "
                       "both y scales are shared, so y_ticks/bar_ticks='outer' keep tick labels on the outer panels, "
                       "and y_titles/bar_titles='outer' do the same for the axis titles.", result)
        self.assertEqual(result.axes.shape, (1, 3))
        self.assertTrue(all(len(ax.lines) == 3 for ax in result.axes.flat))
        self.assertGreater(np.nanmax(result.summary["historical_smm__mean"]), 0.2)  # CPR scale, not SMM
        legend = [text.get_text() for text in result.fig.legends[0].get_texts()]
        self.assertEqual(legend[:3], ["Purpose: Purchase", "Purpose: Rate/Term Refi", "Purpose: Cash-out Refi"])
        self.assertNotIn("CPR", legend)  # one metric needs no line-style entry

    def test_08_matrix_metrics(self):
        result = plot_binned_means(
            self.df,
            x="incentive",
            y=["historical_cpr", "model_cpr"],
            weight="upb",
            bins={"incentive": BinSpec.step(0.25)},
            row="channel",
            col="occupancy",
            bar_mode="total",
            y_scale="shared",
            bar_scale="shared",
            y_format=".0%",
            y_label="CPR (UPB weighted)",
            title="Channel x occupancy",
            labels={"incentive": "Refinance incentive (pp)", "channel": "Channel",
                    "occupancy": "Occupancy", "historical_cpr": "Historical CPR",
                    "model_cpr": "Model CPR"},
        )
        self.save_case("08_matrix_metrics", "matrix", ("多指标 · 统一刻度", "Multiple metrics · shared scales"),
                       "row=... and col=... create a matrix; y_scale='shared' and bar_scale='shared' put every panel on one "
                       "range per axis, while tick labels stay on every panel (the 'all' default).", result)
        self.assertEqual(result.axes.shape, (2, 2))

    def test_09_matrix_groups(self):
        result = plot_binned_means(
            self.df,
            x="incentive",
            y="historical_cpr",
            weight="upb",
            bins={"incentive": BinSpec.step(0.5)},
            group="purpose",
            row="occupancy",
            col="vintage",
            bar_mode="stacked",
            min_count=30,
            y_format=".0%",
            y_label="CPR (UPB weighted)",
            title="CPR by loan purpose · occupancy x vintage",
            labels={"incentive": "Refinance incentive (pp)", "purpose": "Purpose",
                    "occupancy": "Occupancy", "vintage": "Vintage", "historical_cpr": "CPR"},
            y_ticks="outer",
            y_titles="outer",
            bar_titles="outer",
            x_titles="outer",
        )
        self.save_case("09_matrix_groups", "matrix", ("分组叠线", "Groups overlaid"),
                       "group=..., row=... and col=... combine: loan purposes overlay inside an occupancy x vintage matrix. "
                       "Each role splits the rows further, so wider x bins (step 0.5) and min_count=30 keep the thin "
                       "investor cells readable. Count axes stay free (the default) because owner volumes dwarf investor "
                       "ones, so they keep their tick labels while y_ticks='outer' trims the shared y axis.", result)
        self.assertEqual(result.axes.shape, (2, 3))
        self.assertTrue(all(len(ax.lines) == 3 for ax in result.axes.flat))

    def test_10_categorical_points(self):
        result = plot_binned_means(
            self.df,
            x="channel",
            y="historical_cpr",
            weight="upb",
            bins={},  # exact categorical values, no binning
            col="occupancy",
            bar_mode="total",
            y_format=".0%",
            y_label="CPR (UPB weighted)",
            title="Historical CPR by channel",
            labels={"channel": "Channel", "occupancy": "Occupancy",
                    "historical_cpr": "Historical CPR"},
        )
        self.save_case("10_categorical_points", "categorical", ("点图（不连线）· 按列分面", "Unconnected points · column facets"),
                       "Use an unbinned categorical x axis for unconnected metric points; col=... creates one panel per occupancy.", result)


if __name__ == "__main__":
    unittest.main()
