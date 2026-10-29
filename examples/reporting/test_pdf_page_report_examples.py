"""PdfPageReport: one minimal yet complete report that uses each common ingredient once.

``test_minimal_complete_report`` builds one PDF with a title page, sections (with and without a divider page),
a page of text, a table and a chart, a grid of charts that overflows onto a second page, a figure built elsewhere,
and a long table split across pages. The numbers are synthetic; only the layout matters.

Run it with the run button next to the method in the VS Code Test panel (``examples`` must be listed in
``python.testing.pytestArgs``), or from the repository root:

    .venv/Scripts/python.exe -m pytest examples/reporting/test_pdf_page_report_examples.py

The PDF is written to OUTPUT_DIR, which git ignores; open it from there and look at its bookmarks pane too.
"""
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; no plot windows

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from quantbullet.reporting import PdfPageReport  # noqa: E402
from quantbullet.reporting.pdf_text_report import PdfColumnFormat, PdfColumnMeta  # noqa: E402

# Where the example writes its PDF. Change it to write somewhere else.
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "pdf_page_report"


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


class TestPdfPageReportExamples(unittest.TestCase):

    def test_minimal_complete_report(self):
        """Start here: every common page type and block, in one report."""
        rng = np.random.default_rng(0)
        path = output_path("minimal_complete_report.pdf")

        # Leaving the block saves the report; an exception inside it discards the report instead.
        with PdfPageReport(
            path,
            title="Sample Report",                                 # fills {title} in the header and footer
            header=("{title}", "", "{section}"),                   # (left, center, right); one string is left-aligned
            footer=("Synthetic data", "", "Page {page} of {pages}"),  # {pages} is the final page count
        ) as report:
            # A cover page: no header or footer.
            report.add_title_page("Sample Report", subtitle="PdfPageReport ingredients",
                                  notes=["All numbers are synthetic."])

            # ---- A page of blocks: text, table, chart and caption.
            report.add_section("Summary", divider=False)            # no divider: the outline entry is its first page
            with report.page("Fit summary", subtitle="Text, a table, a chart and a caption") as page:
                # Blocks added before chart() stack down from the title ...
                page.text("Paragraphs wrap to the body width. A newline\nstarts a new line.")
                page.bullets(["Bullets wrap with their text aligned.", "Text is plain: $5 is not math."])
                page.heading("Results")
                page.table(
                    pd.DataFrame({"segment": ["A", "B", "C", "Total"],
                                  "count": [1240, 860, 410, 2510],
                                  "share": [0.494, 0.343, 0.163, 1.0]}),
                    schema=[                                         # picks, labels and formats columns; None shows all
                        PdfColumnMeta("segment", "Segment", PdfColumnFormat(align="LEFT")),
                        PdfColumnMeta("count", "Count", PdfColumnFormat(decimals=0, comma=True, align="RIGHT")),
                        PdfColumnMeta("share", "Share", PdfColumnFormat(formatter=lambda v: f"{v:.1%}",
                                                                        align="RIGHT")),
                    ],
                    bold_rows=[-1],                                  # data-row indices; -1 is the Total row
                )
                ax = page.chart()                                    # takes the height left; page.charts(1, 2) for several
                months = np.arange(1, 25)
                ax.plot(months, np.sin(months / 4) + rng.normal(0, 0.1, months.size), label="Actual")
                ax.plot(months, np.sin(months / 4), label="Model")
                ax.set_xlabel("Month")
                ax.legend()
                # ... and blocks added after it stack up from the bottom of the body.
                page.caption("Source: synthetic series.")

            # ---- A grid of charts, filled one Axes at a time.
            report.add_section("Diagnostics", subtitle="A divider page opens this section")
            report.new_page(layout=(2, 2), title="By feature")       # five panels on 2x2 pages: overflows to "By feature (2)"
            for index in range(5):
                ax = report.get_next_ax()
                ax.plot(rng.normal(size=20).cumsum())
                ax.set_title(f"Feature {index}")

            # ---- A figure built elsewhere: its suptitle becomes the page title.
            fig, axes = plt.subplots(1, 2, layout="constrained")
            axes[0].bar(["A", "B", "C"], [3, 5, 2])
            axes[1].scatter(rng.normal(size=50), rng.normal(size=50), s=8)
            fig.suptitle("A figure from elsewhere")
            report.add_figure(fig)                                   # closes the figure

            # ---- A long table on pages of its own, split by rows with the header repeated.
            report.add_section("Appendix", divider=False)
            report.add_table(
                pd.DataFrame({"id": [f"L{i:03d}" for i in range(1, 61)],
                              "balance": rng.uniform(50_000, 750_000, 60).round(-2),
                              "rate": rng.uniform(0.03, 0.08, 60)}),
                title="Detail",                                      # later pages are "Detail (continued)"
                align=["left", "right", "right"],                    # one alignment, or one per column
            )
        print(f"written: {path}")

        reader = PdfReader(path)
        top_level = [item.title for item in reader.outline if not isinstance(item, list)]
        self.assertEqual(top_level, ["Sample Report", "Summary", "Diagnostics", "Appendix"])
        self.assertIn("Detail (continued)", reader.pages[-1].extract_text())


if __name__ == "__main__":
    unittest.main()
