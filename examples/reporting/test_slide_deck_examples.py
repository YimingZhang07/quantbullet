"""SlideDeck on fake prepayment data: one runnable example per test method.

The examples build reveal.js decks in Python with ``quantbullet.reporting.slides`` and render
them with Quarto: a showcase that uses every preset page once, a restyled deck (another
palette, numbered sections, folder output), the built-in "editorial" template, and a page
layout of your own.

``setUpClass`` makes the fake data that every example charts: monthly 1M CPR by vintage.

Run one example with the run button next to its method in the VS Code Test panel (``examples``
must be listed in ``python.testing.pytestArgs``), or from the repository root:

    .venv/Scripts/python.exe -m pytest examples/reporting/test_slide_deck_examples.py -k showcase

Quarto must be installed (on PATH, or QUARTO_PATH pointing at quarto.exe); Plotly charts need
the ``slides`` extra. Decks are written to OUTPUT_DIR, which git ignores; open the .html in a
browser (press ``s`` for the speaker view, ``f`` for full screen).

Known effect in the data: the 2021 vintage's CPR jumps by about 6 points over the last six
months while the other vintages stay flat, so the takeaway and the KPI deltas point at 2021.
"""
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; no plot windows

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402

from quantbullet.reporting import ColumnFormat, ColumnMeta  # noqa: E402
from quantbullet.reporting.slides import (  # noqa: E402
    DEFAULT_THEME,
    ContentSlide,
    SlideDeck,
    find_quarto,
    register_layout,
)

# Where the examples write their decks. Change it to write somewhere else.
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "slide_deck"
VINTAGES = [2019, 2020, 2021, 2022, 2023]
ASOF = "2026-10-31"


def fake_cpr(seed: int) -> pd.DataFrame:
    """Monthly 1M CPR (in %) and balance ($mm) by vintage over 24 months.

    The 2021 vintage's CPR ramps up by about 6 points over the last six months.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(end=ASOF, periods=24, freq="ME")
    rows = []
    for i, vintage in enumerate(VINTAGES):
        base = 6 + 1.5 * i
        ramp = np.where(vintage == 2021, np.clip(np.arange(24) - 17, 0, None) * 1.0, 0.0)
        cpr = base + ramp + rng.normal(0, 0.4, 24)
        balance = 900 * np.exp(-np.cumsum(cpr) / 1200) * (1 + 0.3 * i)
        rows.append(pd.DataFrame({"month": months, "vintage": vintage, "cpr": cpr, "balance": balance}))
    return pd.concat(rows, ignore_index=True)


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def quarto_available() -> bool:
    try:
        find_quarto()
        return True
    except FileNotFoundError:
        return False


@unittest.skipUnless(quarto_available(), "Quarto is not installed")
class TestSlideDeckExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cpr = fake_cpr(seed=0)  # every example charts this

    def test_showcase(self):
        """Start here: every preset page once, in one single-file deck."""
        data = self.cpr
        latest = data[data.month == data.month.max()].set_index("vintage")
        prior = data[data.month == data.month.unique()[-2]].set_index("vintage")

        deck = SlideDeck(
            "RMBS Prepay Monitor",
            subtitle="Synthetic data · October 2026",
            author=["Research"],            # one name or a list
            date=ASOF,
            classification="Internal",      # small tag on the title slide; None hides it
            template="default",             # a registered template name, or a SlideTemplate
            footer="Research · Synthetic data",
            number_sections=False,          # True labels section slides 01, 02, ...
        )

        deck.agenda()                       # no items: lists the sections added below

        deck.section("Market Overview", description="Speeds by vintage")
        with deck.slide("Key numbers") as s:     # a standard slide; layout="content" is the default
            s.kpis([                             # one card per dict (or Kpi objects)
                dict(value=f"{latest.cpr.mean():.1f}", label="Avg 1M CPR",
                     delta=round(latest.cpr.mean() - prior.cpr.mean(), 1)),
                dict(value=f"{latest.loc[2021, 'cpr']:.1f}", label="2021 vintage CPR",
                     delta=round(latest.loc[2021, "cpr"] - prior.loc[2021, "cpr"], 1)),
                dict(value="6.85%", label="Mortgage rate", delta=-0.10,
                     good="down"),               # a fall is good here, so it shows green
            ])
            s.bullets(["Speeds are flat across most vintages.",
                       "The 2021 vintage keeps accelerating."])
            s.source("Synthetic data")           # pinned bottom left
            s.speaker_notes("Mention the 2021 ramp before moving on.")

        deck.takeaway("2021 vintage speeds rose about 6 CPR in six months; other vintages are flat.")

        with deck.theme.matplotlib():            # house palette and slide-sized fonts
            fig, ax = plt.subplots(figsize=(8, 5.5))
            for vintage, group in data.groupby("vintage"):
                ax.plot(group.month, group.cpr, label=str(vintage))
            ax.set_ylabel("1M CPR (%)")
            ax.legend(ncols=5, loc="upper left")
        with deck.two_col("2021 is the outlier", widths=(40, 60)) as s:  # widths in percent
            s.left.bullets(["2021 vintage: +6 CPR since spring.",
                            "2019-2020: flat.",
                            "2022-2023: seasoning slowly."])
            s.left.note("Preliminary; subject to revision")
            s.right.chart(fig)                   # matplotlib -> SVG

        with deck.theme.plotly():                # the theme is Plotly's default template here
            plotly_fig = px.line(data, x="month", y="cpr", color=data.vintage.astype(str),
                                 labels={"cpr": "1M CPR (%)", "month": "", "color": "Vintage"})
        deck.chart_slide("CPR history (interactive)", plotly_fig,
                         height=680)             # Plotly is themed and resized on slide change

        table = latest.reset_index()[["vintage", "cpr", "balance"]]
        table["share"] = table.balance / table.balance.sum()
        total = pd.DataFrame({"vintage": ["Total"], "cpr": [np.average(table.cpr, weights=table.balance)],
                              "balance": [table.balance.sum()], "share": [1.0]})
        table = pd.concat([table.astype({"vintage": str}), total], ignore_index=True)
        with deck.slide("Balances by vintage", layout="dense") as s:   # smaller text for data
            s.table(table, schema=[                          # picks, labels and formats columns
                ColumnMeta("vintage", "Vintage"),
                ColumnMeta("cpr", "1M CPR", ColumnFormat(decimals=1)),
                ColumnMeta("balance", "Balance ($mm)", ColumnFormat(decimals=0, comma=True)),
                ColumnMeta("share", "Share", ColumnFormat(percent=True, decimals=1)),
            ], bold_rows=[-1],                               # the Total row
               caption="Latest month")
            s.markdown("| Plain | Markdown |\n|---|---:|\n| tables | 1.0 |\n| match | 2.0 |")
            s.source("Synthetic data")

        deck.section("Appendix", appendix=True)  # later slides get the Appendix tag
        with deck.slide("Methodology") as s:
            s.text("CPR is the annualized single-month prepayment rate.", size="small")
        deck.disclaimer()                        # the template's partials/disclaimer.md

        path = deck.render(output_path("showcase.html"),
                           embed=True)           # one self-contained file; False writes a folder
        print(f"written: {path}")
        self.assertGreater(path.stat().st_size, 1_000_000)  # plotly.js and fonts are inside

    def test_restyled_deck(self):
        """Another palette, numbered sections, and a folder output with the generated .qmd."""
        data = self.cpr
        deck = SlideDeck(
            "Prepay Monitor",
            date=ASOF,
            theme=DEFAULT_THEME.replace(primary="#141f52", accent="#e3120b",  # colors, fonts
                                        bg_alt="#ebedfa"),
            number_sections=True,
            logo=False,                          # no logo on any slide
        )
        deck.section("Speeds")
        with deck.theme.matplotlib():
            fig, ax = plt.subplots()
            data.pivot_table(index="month", columns="vintage", values="cpr").plot(ax=ax)
            ax.set_ylabel("1M CPR (%)")
        deck.chart_slide("1M CPR by vintage", fig)
        deck.section("Balances")
        deck.takeaway("Balances run off fastest in the 2021 vintage.", title="Runoff")

        path = deck.render(output_path("restyled"), embed=False)  # a folder: index.html, index.qmd, assets
        print(f"written: {path}")
        self.assertTrue((path.parent / "index.qmd").is_file())

    def test_editorial_template(self):
        """The built-in "editorial" template: serif headlines under a red rule, open tables,
        and charts with the value axis on the right."""
        data = self.cpr
        latest = data[data.month == data.month.max()].set_index("vintage")
        deck = SlideDeck(
            "The 2021 vintage breaks away",
            subtitle="Prepayment monitor · October 2026",
            author=["Research"],
            date=ASOF,
            classification="Internal",
            template="editorial",                # its own SlideTheme comes with it
            footer="Research · Synthetic data",
        )
        print(deck.theme)                        # serif headings, red accent, y_axis_side="right"

        deck.agenda()
        deck.section("Speeds", description="Where prepayments are moving")
        with deck.slide("Speeds at a glance") as s:
            s.kpis([dict(value=f"{latest.loc[v, 'cpr']:.1f}", label=f"{v} vintage CPR")
                    for v in (2019, 2021, 2023)])
            s.bullets(["Most vintages are flat.", "The 2021 vintage is up about 6 CPR."])
            s.source("Synthetic data")
        deck.takeaway("Refinancing is concentrated in one vintage, not across the book.")

        with deck.theme.matplotlib():            # value axis on the right, like the slides
            fig, ax = plt.subplots(figsize=(8, 5.5))
            for vintage, group in data.groupby("vintage"):
                ax.plot(group.month, group.cpr, label=str(vintage))
            ax.set_title("1M CPR, %", loc="left")  # units in the title, not on the axis
            ax.legend(ncols=5, loc="upper left")
        with deck.two_col("2021 is the outlier", widths=(40, 60)) as s:
            s.left.bullets(["2021: +6 CPR since spring.", "2019-2020: flat."])
            s.right.chart(fig)
            s.source("Synthetic data")

        with deck.theme.plotly():
            plotly_fig = px.line(data, x="month", y="cpr", color=data.vintage.astype(str),
                                 labels={"cpr": "1M CPR (%)", "month": "", "color": "Vintage"})
        deck.chart_slide("CPR history (interactive)", plotly_fig, height=680)

        table = latest.reset_index()[["vintage", "cpr", "balance"]].astype({"vintage": str})
        with deck.slide("Balances by vintage", layout="dense") as s:
            s.table(table, schema=[
                ColumnMeta("vintage", "Vintage"),
                ColumnMeta("cpr", "1M CPR", ColumnFormat(decimals=1)),
                ColumnMeta("balance", "Balance ($mm)", ColumnFormat(decimals=0, comma=True)),
            ], caption="Latest month")

        deck.section("Appendix", appendix=True)
        deck.disclaimer()

        path = deck.render(output_path("editorial.html"))
        print(f"written: {path}")
        self.assertIn("qb-editorial-title", path.read_text(encoding="utf-8"))

    def test_custom_layout(self):
        """A page layout of your own: a Slide subclass plus a CSS class."""

        @register_layout("scorecard")            # the name is the CSS class on the slide
        class ScorecardSlide(ContentSlide):
            def __init__(self, title=None, scores=None, **kwargs):
                super().__init__(title, **kwargs)
                self.kpis([dict(value=f"{v:.1f}", label=k) for k, v in scores.items()])
                self.html("<style>.reveal section.scorecard .kpi { border-top-color: #2e7d6b; }</style>")

        latest = self.cpr[self.cpr.month == self.cpr.month.max()]
        deck = SlideDeck("Scorecard", date=ASOF)
        deck.slide("Latest CPR by vintage", layout="scorecard",       # found by name ...
                   scores={str(v): c for v, c in zip(latest.vintage, latest.cpr)})
        print(deck.to_qmd().split("---", 2)[2][:400])                 # the generated Markdown
        path = deck.render(output_path("custom_layout.html"))
        print(f"written: {path}")
        self.assertIn("scorecard", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
