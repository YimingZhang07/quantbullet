import datetime as dt
import re
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from quantbullet.reporting.slides import SlideDeck, SlideTheme, TakeawaySlide, find_quarto  # noqa: E402


def quarto_available() -> bool:
    try:
        find_quarto()
        return True
    except FileNotFoundError:
        return False


def sample_deck() -> SlideDeck:
    deck = SlideDeck("Monitor", subtitle="October", author=["A", "B"], date=dt.date(2026, 10, 31),
                     classification="Internal", footer="Research", number_sections=True)
    deck.agenda()
    deck.section("Overview", description="Rates")
    with deck.slide("Key numbers") as s:
        s.kpis([dict(value="12.4", label="1M CPR", delta=2.1)]).source("Synthetic")
    deck.takeaway("One message.")
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [2, 1, 3])
    with deck.two_col("Chart") as s:
        s.left.bullets(["a", "b"])
        s.right.chart(fig)
    deck.slide("Table", layout="dense").table(pd.DataFrame({"x": [1.5, -2.0]}))
    deck.section("Appendix", appendix=True)
    deck.slide("Extra").text("more")
    deck.disclaimer()
    return deck


class TestDeckMarkdown(unittest.TestCase):
    def test_front_matter(self):
        head = sample_deck().to_qmd().split("---")[1]
        self.assertIn('title: "Monitor"', head)
        self.assertIn('author: ["A", "B"]', head)
        self.assertIn('date: "2026-10-31"', head)
        self.assertIn("qb-number-sections: true", head)
        self.assertIn('"qb-default-revealjs": {"footer": "Research"', head)
        self.assertNotIn("embed-resources", head)
        self.assertIn('"embed-resources": true', sample_deck().to_qmd(embed=True))

    def test_slides_in_order_with_appendix_tags(self):
        qmd = sample_deck().to_qmd()
        headings = re.findall(r"^#+ .*$", qmd, flags=re.M)
        self.assertEqual(headings, [
            "## Agenda {.agenda}", "# Overview", "## Key numbers", "## {.takeaway}",
            "## Chart {.two-col}", "## Table {.dense}", "# Appendix {.appendix-section}",
            "## Extra {.appendix}", "## Disclaimer {.dense .disclaimer}",
        ])
        self.assertIn("1. Overview\n", qmd)  # the agenda lists non-appendix sections only

    def test_slide_by_layout_name_and_add(self):
        deck = SlideDeck("x")
        two = deck.slide("T", layout="two-col", widths=(30, 70))
        self.assertEqual(two.layout, "two-col")
        takeaway = deck.add(TakeawaySlide("m"))
        self.assertIs(deck.slides[-1], takeaway)
        with self.assertRaises(TypeError):
            deck.add("not a slide")
        with self.assertRaises(TypeError):
            deck.title_slide(colour="red")

    def test_logo_off_and_theme(self):
        deck = SlideDeck("x", logo=False, theme=SlideTheme(primary="#141f52"),
                         options={"slide-number": False})
        head = deck.to_qmd().split("---")[1]
        self.assertNotIn("logo", head)
        self.assertIn('qb-section-background: "#141f52"', head)
        self.assertIn('"slide-number": false', head)

    def test_build_writes_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            qmd = sample_deck().build(tmp)
            self.assertEqual(qmd, Path(tmp) / "index.qmd")
            self.assertTrue((Path(tmp) / "assets" / "fig-001.svg").is_file())
            self.assertTrue((Path(tmp) / "_extensions" / "qb-default" / "_extension.yml").is_file())
            self.assertTrue((Path(tmp) / "_quarto.yml").is_file())


@unittest.skipUnless(quarto_available(), "Quarto is not installed")
class TestDeckRender(unittest.TestCase):
    def test_render_single_file(self):
        import plotly.graph_objects as go

        deck = sample_deck()
        deck.chart_slide("Interactive", go.Figure(go.Scatter(y=[1, 3, 2])))
        with tempfile.TemporaryDirectory() as tmp:
            out = deck.render(Path(tmp) / "deck.html")
            html = out.read_text(encoding="utf-8")
        self.assertIn('class="qb-title', html)
        self.assertIn("qb-title", html)
        self.assertIn("section-slide", html)
        self.assertIn('data-background-color="#0b3d6e"', html)
        self.assertIn('class="kpi-value"', html)
        self.assertIn("plotly-graph-div", html)
        self.assertIn("section-num", html)
        self.assertIsNone(re.search(r'<script src="(?!data:)[^"]+"', html))  # nothing external

    def test_render_editorial(self):
        deck = SlideDeck("Editorial", template="editorial")
        deck.section("Part one")
        deck.takeaway("One message.")
        with tempfile.TemporaryDirectory() as tmp:
            html = deck.render(Path(tmp) / "deck.html").read_text(encoding="utf-8")
        self.assertIn("qb-editorial-title", html)
        self.assertIn('data-background-color="#121212"', html)  # section slide
        self.assertIn("Source Serif 4", html)

    def test_render_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = sample_deck().render(Path(tmp) / "deck", embed=False)
            self.assertEqual(out.name, "index.html")
            self.assertTrue((out.parent / "assets" / "fig-001.svg").is_file())


if __name__ == "__main__":
    unittest.main()
