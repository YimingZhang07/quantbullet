import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.textpath import text_to_path
from pypdf import PdfReader

from quantbullet.plot.theme import PRINT_RC
from quantbullet.reporting import PdfPageReport, ReportPage, ReportStyle
from quantbullet.reporting._reportlab_utils import PdfColumnFormat, PdfColumnMeta
from tests.artifacts import artifact_dir


class TestPdfPageReport(unittest.TestCase):
    def setUp(self):
        self.output_dir = artifact_dir(self, "reporting/pdf_page_report")

    def _path(self, name):
        return Path(self.output_dir) / name

    def _line_figure(self, title, **kwargs):
        fig, ax = plt.subplots(figsize=(4, 3), **kwargs)
        ax.plot([0, 1, 2], [1, 3, 2])
        ax.set_title(title)
        return fig

    def _assert_page_size(self, reader, width=11 * 72, height=8.5 * 72):
        for page in reader.pages:
            self.assertAlmostEqual(float(page.mediabox.width), width)
            self.assertAlmostEqual(float(page.mediabox.height), height)

    def _ink(self, fig, artist):
        """An artist's drawn extent in figure fractions."""
        return artist.get_tightbbox(fig._get_renderer()).transformed(fig.transFigure.inverted())

    def _assert_in_body(self, fig, artists, style=ReportStyle()):
        width, height = style.page_size
        left, right, top, bottom = style.margins
        for artist in artists:
            box = self._ink(fig, artist)
            self.assertGreaterEqual(box.x0, left / width - 1e-3)
            self.assertLessEqual(box.x1, 1 - right / width + 1e-3)
            self.assertGreaterEqual(box.y0, bottom / height - 1e-3)
            self.assertLessEqual(box.y1, 1 - top / height + 1e-3)

    def test_sections_outline_order_and_full_figure_size(self):
        path = self._path("sections.pdf")
        report = PdfPageReport(path)
        report.new_page(layout=(1, 1), title="Introduction")
        report.get_next_ax().plot([0, 1], [0, 1])

        report.add_section("First Section")
        report.new_page(layout=(1, 1), title="Grid Title", bookmark_title="Grid Bookmark")
        report.get_next_ax().plot([0, 1], [1, 0])
        report.add_figure(self._line_figure("Vector Chart"), title="Full Figure")

        report.add_section("Second Section")
        report.add_figure(self._line_figure("Last Chart"))
        report.save()

        reader = PdfReader(path)
        self.assertEqual(len(reader.pages), 6)
        self._assert_page_size(reader)
        self.assertIn("First Section", reader.pages[1].extract_text())
        self.assertIn("Grid Title", reader.pages[2].extract_text())
        self.assertIn("Vector Chart", reader.pages[3].extract_text())
        self.assertIn("Second Section", reader.pages[4].extract_text())
        self.assertIn("Last Chart", reader.pages[5].extract_text())

        outline = reader.outline
        self.assertEqual(len(outline), 5)
        self.assertEqual([outline[0].title, outline[1].title, outline[3].title],
                         ["Introduction", "First Section", "Second Section"])
        self.assertEqual([item.title for item in outline[2]],
                         ["Grid Bookmark", "Full Figure"])
        self.assertEqual([item.title for item in outline[4]], ["Page 6"])
        for item, page_index in [
            (outline[0], 0), (outline[1], 1), (outline[2][0], 2),
            (outline[2][1], 3), (outline[3], 4), (outline[4][0], 5),
        ]:
            self.assertEqual(reader.get_destination_page_number(item), page_index)

    def test_automatic_overflow_and_empty_page_requests(self):
        path = self._path("overflow.pdf")
        report = PdfPageReport(path, layout=(1, 1))
        report.new_page(title="Unused")
        report.new_page(title="Plots")
        report.get_next_ax().plot([0, 1], [0, 1])
        report.get_next_ax().plot([0, 1], [1, 0])
        report.add_pagebreak()
        report.new_page(title="Also unused")
        report.save()

        reader = PdfReader(path)
        self.assertEqual(len(reader.pages), 2)
        self.assertEqual([item.title for item in reader.outline], ["Plots", "Plots (2)"])
        self.assertIn("Plots (2)", reader.pages[1].extract_text())

    def test_mixed_grid_and_figure_preserves_call_order(self):
        path = self._path("mixed.pdf")
        report = PdfPageReport(path, layout=(1, 2))
        report.new_page(title="Grid First")
        report.get_next_ax().set_title("First Grid Plot")
        report.add_figure(self._line_figure("Middle Figure"), bookmark_title="Middle")
        report.new_page(layout=(1, 1), title="Grid Last")
        report.get_next_ax().set_title("Last Grid Plot")
        report.save()

        reader = PdfReader(path)
        self.assertEqual(len(reader.pages), 3)
        self.assertEqual([item.title for item in reader.outline],
                         ["Grid First", "Middle", "Grid Last"])
        self.assertIn("First Grid Plot", reader.pages[0].extract_text())
        self.assertIn("Middle Figure", reader.pages[1].extract_text())
        self.assertIn("Last Grid Plot", reader.pages[2].extract_text())
        self._assert_page_size(reader)

    def test_external_axes_and_chrome_stay_vector(self):
        path = self._path("vector.pdf")
        source_fig = self._line_figure("Source Plot")
        report = PdfPageReport(path, layout=(1, 1), footer="Page {page}")
        report.add_external_axes([source_fig.axes[0]])
        report.save()
        plt.close(source_fig)

        reader = PdfReader(path)
        self.assertEqual(len(reader.pages), 1)
        self.assertEqual(reader.outline[0].title, "Page 1")
        self.assertIn("Source Plot", reader.pages[0].extract_text())
        self.assertIn("Page 1", reader.pages[0].extract_text())
        xobjects = reader.pages[0]["/Resources"].get("/XObject", {})
        self.assertFalse(any(obj.get_object().get("/Subtype") == "/Image"
                             for obj in xobjects.values()))

    def test_page_blocks_stack_around_the_chart(self):
        path = self._path("blocks.pdf")
        report = PdfPageReport(path)
        with report.page("Summary", subtitle="Subtitle line") as page:
            page.text("Intro paragraph")
            page.heading("Counts")
            page.table(pd.DataFrame({"name": ["a", "b"], "count": [1234, 5678]}))
            page.bullets(["First point", "Second point"])
            ax = page.chart()
            ax.plot([0, 1], [0, 1])
            ax.set_xlabel("x label")
            page.caption("Source line")
        report.save()

        fig = page.fig
        host = next(other for other in fig.axes if other is not ax)
        caption = next(text for text in fig.texts if text.get_text() == "Source line")
        bullet = next(text for text in fig.texts if text.get_text() == "Second point")
        chart = self._ink(fig, ax)
        self.assertGreater(self._ink(fig, host).y0, self._ink(fig, bullet).y1)
        self.assertLess(chart.y1, self._ink(fig, bullet).y0)
        self.assertLess(self._ink(fig, caption).y1, chart.y0)
        self._assert_in_body(fig, [ax, host, caption, *fig.texts])

        text = PdfReader(path).pages[0].extract_text()
        for expected in ["Summary", "Subtitle line", "Intro paragraph", "Counts", "1,234",
                         "Second point", "x label", "Source line"]:
            self.assertIn(expected, text)

    def test_text_wraps_to_the_body_width(self):
        path = self._path("wrap.pdf")
        style = ReportStyle()
        report = PdfPageReport(path, style=style)
        with report.page() as page:
            page.text("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 8)
            page.bullets(["Ut enim ad minim veniam, quis nostrud exercitation. " * 6])
        report.save()

        body = style.page_size[0] - style.margins[0] - style.margins[1]
        paragraph, item = (text for text in page.fig.texts if text.get_text() != "•")
        for artist, width in [(paragraph, body), (item, body - 0.18)]:
            lines = artist.get_text().split("\n")
            self.assertGreater(len(lines), 1)
            for line in lines:
                line_width = text_to_path.get_text_width_height_descent(
                    line, artist.get_fontproperties(), ismath=False)[0] / 72
                self.assertLessEqual(line_width, width)

    def test_schema_table_formats_labels_and_aligns(self):
        path = self._path("table.pdf")
        frame = pd.DataFrame({"item": ["Alpha", "Total"], "count": [1240.0, 2730.0],
                              "share": [0.924, None], "hidden": [1, 2]})
        schema = [
            PdfColumnMeta("item", "Item", PdfColumnFormat(align="LEFT")),
            PdfColumnMeta("count", "Count", PdfColumnFormat(decimals=0, comma=True, align="RIGHT")),
            PdfColumnMeta("share", "Share", PdfColumnFormat(formatter=lambda v: f"{v:.1%}",
                                                            missing_label="n/a")),
        ]
        report = PdfPageReport(path)
        with report.page("Tables") as page:
            page.table(frame, schema=schema, col_widths=[2, 1, 1], bold_rows=[-1])
            page.table([["Rows", 12], ["Ratio", 0.5]], align=["left", "right"])
        report.save()

        schema_table, plain_table = (ax.tables[0] for ax in page.fig.axes)
        cells = schema_table.get_celld()
        self.assertEqual([cells[(0, col)].get_text().get_text() for col in range(3)],
                         ["Item", "Count", "Share"])
        self.assertEqual([cells[(2, col)].get_text().get_text() for col in range(3)],
                         ["Total", "2,730", "n/a"])
        self.assertEqual(cells[(1, 2)].get_text().get_text(), "92.4%")
        self.assertEqual(cells[(1, 0)].get_text().get_horizontalalignment(), "left")
        self.assertEqual(cells[(1, 1)].get_text().get_horizontalalignment(), "right")
        self.assertEqual(cells[(2, 0)].get_text().get_fontweight(), "bold")
        self.assertAlmostEqual(cells[(1, 0)].get_width(), 2 * cells[(1, 1)].get_width())
        # Rows without columns have no header row.
        self.assertEqual(plain_table.get_celld()[(0, 0)].get_text().get_text(), "Rows")
        self.assertNotIn("hidden", PdfReader(path).pages[0].extract_text())

    def test_header_footer_fields_and_uncounted_cover_and_dividers(self):
        path = self._path("chrome.pdf")
        report = PdfPageReport(path, title="Doc", header=("{title}", "", "{section}"),
                                footer=("", "", "Page {page} of {pages}"))
        report.add_title_page("Cover Title", subtitle="Cover subtitle", notes=["A note"])
        report.add_section("Alpha", divider=False)
        report.page("One").close()
        report.add_section("Beta")
        with report.page("Two"):
            pass
        report.save()

        reader = PdfReader(path)
        texts = [page.extract_text() for page in reader.pages]
        self.assertEqual(len(texts), 4)
        self.assertIn("Cover Title", texts[0])
        self.assertNotIn("Page 1 of 4", texts[0])
        self.assertIn("Page 2 of 4", texts[1])
        self.assertIn("Alpha", texts[1])
        self.assertIn("Doc", texts[1])
        self.assertNotIn("Page 3 of 4", texts[2])
        self.assertIn("Page 4 of 4", texts[3])
        self.assertIn("Beta", texts[3])

        outline = reader.outline
        self.assertEqual([outline[0].title, outline[1].title, outline[3].title],
                         ["Cover Title", "Alpha", "Beta"])
        self.assertEqual([item.title for item in outline[2]], ["One"])
        self.assertEqual([item.title for item in outline[4]], ["Two"])
        self.assertEqual(reader.get_destination_page_number(outline[1]), 1)
        self.assertEqual(reader.get_destination_page_number(outline[4][0]), 3)
        self.assertEqual(reader.metadata.title, "Doc")
        self.assertEqual(reader.page_mode, "/UseOutlines")

    def test_overflowing_pages_raise_and_the_report_continues(self):
        path = self._path("too_tall.pdf")
        report = PdfPageReport(path)
        with self.assertRaisesRegex(ValueError, "past the bottom of the body"):
            with report.page("Long") as page:
                page.table([[str(row)] for row in range(200)])
        with self.assertRaisesRegex(ValueError, "for its charts"):
            with report.page("Squeezed") as page:
                page.table([[str(row)] for row in range(20)])
                page.chart()
        with report.page("Fine") as page:
            with self.assertRaisesRegex(ValueError, "one chart region"):
                page.chart()
                page.charts(1, 2)
        report.save()

        reader = PdfReader(path)
        self.assertEqual([item.title for item in reader.outline], ["Fine"])
        with self.assertRaisesRegex(RuntimeError, "already been written"):
            page.text("late")

    def test_section_without_pages_and_bad_templates_raise(self):
        with self.assertRaisesRegex(ValueError, "Bad footer template"):
            PdfPageReport(self._path("unused.pdf"), footer="{pagez}")

        path = self._path("empty_section.pdf")
        report = PdfPageReport(path)
        report.page("Kept").close()
        report.add_section("Empty", divider=False)
        with self.assertRaisesRegex(ValueError, "'Empty' has no pages"):
            report.save()
        self.assertFalse(path.exists())

    def test_added_figures_fit_the_body_below_the_title(self):
        path = self._path("added.pdf")
        constrained = self._line_figure("Left", layout="constrained")
        constrained.axes[0].set_ylabel("A long y label")
        constrained.suptitle("Own Suptitle")
        plain = self._line_figure("Plain chart")
        report = PdfPageReport(path)
        report.add_figure(constrained)
        report.add_figure(plain, title="Plain")
        report.save()

        reader = PdfReader(path)
        self.assertEqual([item.title for item in reader.outline], ["Own Suptitle", "Plain"])
        self.assertEqual(constrained.get_suptitle(), "")
        for fig in (constrained, plain):
            title = next(text for text in fig.texts if text.get_text() in ("Own Suptitle", "Plain"))
            self._assert_in_body(fig, [*fig.axes, title])
            self.assertLess(self._ink(fig, fig.axes[0]).y1, self._ink(fig, title).y0)

    def test_report_rc_applies_inside_page_blocks_only(self):
        path = self._path("rc.pdf")
        report = PdfPageReport(path, layout=(1, 1))
        with report.page() as page:
            page_ax = page.chart()
            page_ax.set_title("In block")
        grid_ax = report.get_next_ax()
        grid_ax.set_title("Grid")
        report.save()

        reference_fig, reference_ax = plt.subplots()
        reference_ax.set_title("Reference")
        self.assertEqual(page_ax.title.get_fontsize(), PRINT_RC["axes.titlesize"])
        self.assertEqual(grid_ax.title.get_fontsize(), reference_ax.title.get_fontsize())
        self.assertEqual(grid_ax.xaxis.get_ticklabels()[0].get_fontsize(),
                         reference_ax.xaxis.get_ticklabels()[0].get_fontsize())
        plt.close(reference_fig)

    def _leftovers(self):
        return list(Path(self.output_dir).glob(".pdf_page_*"))

    def test_context_manager_saves_or_discards_the_report(self):
        saved = self._path("with_saved.pdf")
        with PdfPageReport(saved) as report:
            report.page("Kept").close()
        self.assertEqual(len(PdfReader(saved).pages), 1)
        with PdfPageReport(self._path("saved_inside.pdf")) as report:
            report.page("Kept").close()
            report.save()

        failed = self._path("with_failed.pdf")
        with self.assertRaisesRegex(KeyError, "boom"):
            with PdfPageReport(failed) as report:
                report.page("Written").close()
                report.page("Open").text("Never written")
                raise KeyError("boom")
        self.assertFalse(failed.exists())
        with self.assertRaisesRegex(RuntimeError, "saved or discarded"):
            report.page()

        manual = PdfPageReport(self._path("manual.pdf"))
        manual.page("Dropped").close()
        manual.discard()
        manual.discard()
        self.assertFalse(self._path("manual.pdf").exists())
        self.assertEqual(self._leftovers(), [])

    def test_add_table_splits_rows_and_repeats_the_header(self):
        path = self._path("long_table.pdf")
        codes = [f"R{index:03d}" for index in range(120)]
        calls = []
        table_block = ReportPage._table_block

        def spy(page, parts, rows, bold, **kwargs):
            calls.append((rows, bold))
            return table_block(page, parts, rows, bold, **kwargs)

        report = PdfPageReport(path)
        with patch.object(ReportPage, "_table_block", spy):
            report.add_table(pd.DataFrame({"code": codes, "value": range(120)}), title="Loan detail",
                             subtitle="Every loan", bold_rows=[0, -1])
            report.add_table([[f"K{index:02d}", index] for index in range(5)])
        report.save()

        loan_calls, plain_calls = calls[:-1], calls[-1:]
        self.assertGreater(len(loan_calls), 2)
        self.assertEqual([row[0] for rows, _ in loan_calls for row in rows], codes)
        sizes = [len(rows) for rows, _ in loan_calls]
        self.assertLess(sizes[0], sizes[1])  # the subtitle takes room on the first page only
        self.assertEqual(len(set(sizes[1:-1])), 1)
        self.assertEqual([bold for _, bold in loan_calls],
                         [{0}] + [set()] * (len(loan_calls) - 2) + [{sizes[-1] - 1}])
        self.assertEqual([row[0] for row in plain_calls[0][0]], [f"K{index:02d}" for index in range(5)])

        reader = PdfReader(path)
        self.assertEqual([item.title for item in reader.outline],
                         ["Loan detail"] + ["Loan detail (continued)"] * (len(loan_calls) - 1)
                         + [f"Page {len(loan_calls) + 1}"])
        texts = [page.extract_text() for page in reader.pages]
        for text in texts[:-1]:
            self.assertIn("code", text)
            self.assertIn("value", text)
        self.assertIn("Every loan", texts[0])
        self.assertNotIn("Every loan", texts[1])
        self.assertNotIn("code", texts[-1])

    def test_add_table_rejects_rows_taller_than_the_body(self):
        report = PdfPageReport(self._path("huge_font.pdf"))
        with self.assertRaisesRegex(ValueError, "No table row fits"):
            report.add_table([["a"]], columns=["Header"], font_size=200)
        report.discard()
        self.assertEqual(self._leftovers(), [])


if __name__ == "__main__":
    unittest.main()
