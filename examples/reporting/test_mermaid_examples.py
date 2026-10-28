"""Mermaid diagram files rendered to SVG and PDF: one runnable example per test method.

These examples use ``quantbullet.reporting.render_mermaid`` the way a user would. Start with
``test_minimal_complete_flowchart``: it writes its own small flowchart, which uses each common flowchart feature
once, and renders it to SVG and PDF. ``setUpClass`` runs once and writes the two Mermaid source files the other
examples read: a model-build flowchart and a sequence diagram. Each of those renders one file with different
options.

Run one example with the run button next to its method in the VS Code Test panel (``examples`` must be listed
in ``python.testing.pytestArgs``); ``setUpClass`` runs first, then that method. From the repository root:

    .venv/Scripts/python.exe -m pytest examples/reporting/test_mermaid_examples.py -k minimal_complete_flowchart

The sources and the rendered SVG / PDF files are written to OUTPUT_DIR, which git ignores; open them from there.
Rendering runs mermaid.js in a headless browser and loads it from a CDN, so the examples need network access.
Each one takes a few seconds, most of it starting the browser.
"""
import unittest
from pathlib import Path

from pypdf import PdfReader

from quantbullet.reporting import render_mermaid

# Where the examples write their files. Change it to write somewhere else.
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "mermaid"

# The browser that runs mermaid.js: "msedge" uses an installed Edge, "chrome" an installed Chrome, and None
# uses Playwright's own Chromium, which needs a one-off `playwright install chromium`.
BROWSER_CHANNEL = "msedge"

# A small flowchart that still uses each common feature once: a title, comments, node shapes, labelled solid /
# dotted / thick links, a line break in a label, a subgraph with its own direction, and a style class.
FLOWCHART_TOUR = """\
---
title: Prepayment model build
---
flowchart TD
    %% shapes: [(database)] [rectangle] {decision} [[subroutine]] (rounded) {{hexagon}} [/parallelogram/] ([stadium])
    tape[(Loan tape)] --> clean[Clean and filter]
    clean --> enough{Enough<br/>history?}
    enough -- yes --> fit[[Fit GAM]]
    enough -. no .-> pool(Pool with similar loans)
    pool --> fit

    subgraph checks [Diagnostics]
        direction LR
        ae{{A/E by bucket}} --> twoway[/Two-way correction/]
    end

    fit ==> checks
    checks --> report([PDF report])

    classDef model fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px
    class fit,pool model
"""

MODEL_FLOWCHART = """\
flowchart LR
    raw[(Loan tape)] --> clean[Clean and filter]
    clean --> check{Enough<br/>history?}
    check -- yes --> fit[Fit GAM]
    check -- no --> pool[Pool with<br/>similar loans]
    pool --> fit
    fit --> ae[A/E diagnostics]
    ae --> report[[PDF report]]
"""

SCORING_SEQUENCE = """\
sequenceDiagram
    participant U as Analyst
    participant M as Model
    participant D as Database
    U->>D: query loan tape
    D-->>U: loans
    U->>M: predict(loans)
    M-->>U: SMM per loan
    U->>U: compare with actuals
"""


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


class TestMermaidExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Every example renders one of these source files, as a user would render their own .mmd files.
        cls.flowchart = output_path("model_flow.mmd")
        cls.flowchart.write_text(MODEL_FLOWCHART, encoding="utf-8")
        cls.sequence = output_path("scoring_sequence.mmd")
        cls.sequence.write_text(SCORING_SEQUENCE, encoding="utf-8")

    def test_minimal_complete_flowchart(self):
        """Start here: write a flowchart to a .mmd file, then render it to both SVG and PDF."""
        source = output_path("flowchart_tour.mmd")
        source.write_text(FLOWCHART_TOUR, encoding="utf-8")

        svg = render_mermaid(
            source,
            output_path("flowchart_tour.svg"),    # the suffix picks the format: ".svg" or ".pdf"
            theme="default",                      # "default", "neutral", "dark", "forest" or "base"
            background="white",                   # any CSS color; "transparent" leaves it see-through
            browser_channel=BROWSER_CHANNEL,      # see BROWSER_CHANNEL above
        )
        pdf = render_mermaid(
            source,
            output_path("flowchart_tour.pdf"),    # one page, sized to the diagram
            theme="default",
            background="white",
            browser_channel=BROWSER_CHANNEL,
        )
        print(f"written: {svg}\nwritten: {pdf}")

        self.assertIn("Two-way correction", svg.read_text(encoding="utf-8"))
        pages = PdfReader(pdf).pages
        self.assertEqual(len(pages), 1)
        self.assertIn("Diagnostics", pages[0].extract_text())

    def test_flowchart_to_svg(self):
        """The simplest call: the output suffix picks the format."""
        path = render_mermaid(
            self.flowchart,
            output_path("model_flow.svg"),        # ".svg" writes SVG; ".pdf" writes a one-page PDF
            browser_channel=BROWSER_CHANNEL,      # see BROWSER_CHANNEL above
        )
        print(f"written: {path}")
        self.assertIn("Fit GAM", path.read_text(encoding="utf-8"))

    def test_flowchart_to_pdf(self):
        """A PDF page sized to the diagram, with a quieter theme for print."""
        path = render_mermaid(
            self.flowchart,
            output_path("model_flow.pdf"),
            theme="neutral",                      # "default", "neutral", "dark", "forest" or "base"
            background="white",                   # any CSS color; "transparent" leaves it see-through
            browser_channel=BROWSER_CHANNEL,
        )
        page = PdfReader(path).pages[0]
        print(f"written: {path}  ({float(page.mediabox.width):.0f} x {float(page.mediabox.height):.0f} pt)")
        self.assertEqual(len(PdfReader(path).pages), 1)

    def test_svg_for_non_browser_tools(self):
        """Labels as plain SVG text, so Inkscape, cairosvg and similar tools can show them."""
        path = render_mermaid(
            self.flowchart,
            output_path("model_flow_plain_text.svg"),
            config={"htmlLabels": False},         # any mermaid.initialize setting; default labels are HTML
            browser_channel=BROWSER_CHANNEL,
        )
        print(f"written: {path}")
        self.assertNotIn("<foreignObject", path.read_text(encoding="utf-8"))

    def test_sequence_diagram_to_pdf(self):
        """Any Mermaid diagram type works, not only flowcharts."""
        path = render_mermaid(
            self.sequence,
            output_path("scoring_sequence.pdf"),
            theme="default",
            background="white",
            browser_channel=BROWSER_CHANNEL,
        )
        print(f"written: {path}")
        self.assertIn("predict(loans)", PdfReader(path).pages[0].extract_text())


if __name__ == "__main__":
    unittest.main()
