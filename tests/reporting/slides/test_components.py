import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from quantbullet.reporting.columns import ColumnFormat, ColumnMeta, ColumnSchema  # noqa: E402
from quantbullet.reporting.pdf_text_report import PdfColumnFormat, PdfColumnMeta  # noqa: E402
from quantbullet.reporting.slides import (  # noqa: E402
    Bullets,
    Column,
    Columns,
    Figure,
    Kpi,
    KpiRow,
    Note,
    PlotlyFigure,
    RenderContext,
    Source,
    Table,
    Text,
)


class TestTextComponents(unittest.TestCase):
    def setUp(self):
        self.ctx = RenderContext()

    def test_text_sizes(self):
        self.assertEqual(Text("hi").to_markdown(self.ctx), "hi")
        self.assertIn(".qb-text-small", Text("hi", size="small").to_markdown(self.ctx))
        with self.assertRaises(ValueError):
            Text("hi", size="huge")

    def test_bullets(self):
        self.assertEqual(Bullets(["a", "b"]).to_markdown(self.ctx), "- a\n- b")
        self.assertEqual(Bullets(["a", "b"], numbered=True).to_markdown(self.ctx), "1. a\n2. b")
        self.assertIn(".incremental", Bullets(["a"], incremental=True).to_markdown(self.ctx))

    def test_user_text_is_escaped_in_html(self):
        self.assertIn("Source: A &amp; B &lt;x&gt;", Source("A & B <x>").to_markdown(self.ctx))
        self.assertIn("&lt;b&gt;", Note("<b>").to_markdown(self.ctx))


class TestKpi(unittest.TestCase):
    def test_delta_tone_follows_good_direction(self):
        self.assertIn('kpi-delta good">▲ +2.1', Kpi("12.4", "CPR", delta=2.1).html())
        self.assertIn('kpi-delta bad">▼ -0.1', Kpi("1", "x", delta=-0.1).html())
        self.assertIn('kpi-delta good">▼ -0.1', Kpi("1", "x", delta=-0.1, good="down").html())
        self.assertIn('kpi-delta flat">+0', Kpi("1", "x", delta=0).html())
        self.assertIn('kpi-delta bad">▲ +3 bp', Kpi("1", "x", delta="+3 bp", good="down").html())
        self.assertNotIn("kpi-delta", Kpi("1", "x").html())

    def test_numpy_delta(self):
        import numpy as np

        self.assertIn('kpi-delta good">▲ +0.5', Kpi("1", "x", delta=np.float64(0.5)).html())

    def test_row_accepts_dicts(self):
        md = KpiRow([dict(value="1", label="a"), Kpi("2", "b")]).to_markdown(RenderContext())
        self.assertEqual(md.count('class="kpi"'), 2)


class TestTable(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"name": ["A", "B", "Total"], "bal": [1234.5, -20.0, 1214.5],
                                "share": [0.5, None, 1.0]})

    def test_default_formats_and_alignment(self):
        html = Table(self.df).html
        self.assertIn('<th class="num">bal</th>', html)
        self.assertIn('<td class="num neg">-20</td>', html)
        self.assertIn("<td>A</td>", html)

    def test_schema_picks_labels_and_formats(self):
        schema = ColumnSchema([
            ColumnMeta("name", "Name"),
            ColumnMeta("share", "Share", ColumnFormat(percent=True, decimals=1)),
            ColumnMeta("bal", "Balance", ColumnFormat(decimals=0, comma=True, parens_for_negative=True)),
        ])
        html = Table(self.df, schema=schema, bold_rows=[-1]).html
        self.assertLess(html.index(">Share<"), html.index(">Balance<"))
        self.assertIn(">50.0%<", html)
        self.assertIn(">(20)<", html)
        self.assertIn('<tr class="bold"><td>Total</td>', html)

    def test_pdf_schema_formatter_and_missing_label(self):
        schema = [PdfColumnMeta("share", "Share", PdfColumnFormat(formatter=lambda v: f"{v:.0%}",
                                                                  missing_label="n/a", align="CENTER"))]
        html = Table(self.df, schema=schema).html
        self.assertIn(">50%<", html)
        self.assertIn(">n/a<", html)
        self.assertIn("text-align:center", html)

    def test_guards(self):
        with self.assertRaisesRegex(ValueError, "max_rows"):
            Table(self.df, max_rows=2)
        with self.assertRaises(KeyError):
            Table(self.df, schema=[ColumnMeta("nope")])
        with self.assertRaises(IndexError):
            Table(self.df, bold_rows=[5])
        with self.assertRaises(TypeError):
            Table([[1, 2]])


class TestCharts(unittest.TestCase):
    def test_matplotlib_figure_is_written_as_asset(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        component = Figure(ax, width="80%")  # an Axes is accepted too
        self.assertIn(b"<svg", component.data)
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RenderContext(assets_dir=Path(tmp))
            md = component.to_markdown(ctx)
            self.assertEqual(md, '![](assets/fig-001.svg){width="80%"}')
            self.assertTrue((Path(tmp) / "fig-001.svg").is_file())

    def test_plotly_script_is_loaded_once(self):
        import plotly.graph_objects as go

        fig = go.Figure(go.Scatter(y=[1, 2, 3]))
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RenderContext(assets_dir=Path(tmp))
            first = PlotlyFigure(fig).to_markdown(ctx)
            second = PlotlyFigure(fig, height=300).to_markdown(ctx)
            self.assertIn('<script src="assets/plotly.min.js">', first)
            self.assertNotIn("plotly.min.js", second)
            self.assertIn("height:300px", second)
            self.assertIn('"displaylogo": false', first)
            self.assertTrue((Path(tmp) / "plotly.min.js").is_file())
        # The caller's figure keeps Plotly's default template; only the rendered copy is styled.
        self.assertNotEqual(fig.layout.template.layout.colorway[0], ctx.theme.primary)
        self.assertIn(ctx.theme.primary, first)


class TestColumns(unittest.TestCase):
    def test_columns_markdown(self):
        left, right = Column().text("L"), Column().text("R")
        md = Columns(left, right, widths=(40, 60)).to_markdown(RenderContext())
        self.assertTrue(md.startswith(":::: {.columns}"))
        self.assertIn('::: {.column width="40%"}\nL\n:::', md)
        with self.assertRaises(ValueError):
            Columns(left, right, widths=(100,))


if __name__ == "__main__":
    unittest.main()
