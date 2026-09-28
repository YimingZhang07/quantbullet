import unittest
from pathlib import Path

import matplotlib.pyplot as plt
from pypdf import PdfReader

from quantbullet.reporting import PdfChartReport
from tests.artifacts import artifact_dir


class TestPdfChartReport(unittest.TestCase):
    def setUp(self):
        self.output_dir = artifact_dir(self, "reporting/pdf_chart_report")

    def _path(self, name):
        return Path(self.output_dir) / name

    def _line_figure(self, title):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1, 2], [1, 3, 2])
        ax.set_title(title)
        return fig

    def _assert_page_size(self, reader, width=11 * 72, height=8.5 * 72):
        for page in reader.pages:
            self.assertAlmostEqual(float(page.mediabox.width), width)
            self.assertAlmostEqual(float(page.mediabox.height), height)

    def test_sections_outline_order_and_full_figure_size(self):
        path = self._path("sections.pdf")
        report = PdfChartReport(path)
        report.new_page(layout=(1, 1), suptitle="Introduction")
        report.get_next_ax().plot([0, 1], [0, 1])

        report.add_section("First Section")
        report.new_page(layout=(1, 1), suptitle="Grid Title", bookmark_title="Grid Bookmark")
        report.get_next_ax().plot([0, 1], [1, 0])
        report.add_figure(self._line_figure("Vector Chart"), suptitle="Full Figure")

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
        report = PdfChartReport(path, layout=(1, 1))
        report.new_page(suptitle="Unused")
        report.new_page(suptitle="Plots")
        report.get_next_ax().plot([0, 1], [0, 1])
        report.get_next_ax().plot([0, 1], [1, 0])
        report.add_pagebreak()
        report.new_page(suptitle="Also unused")
        report.save()

        reader = PdfReader(path)
        self.assertEqual(len(reader.pages), 2)
        self.assertEqual([item.title for item in reader.outline], ["Plots", "Plots (2)"])
        self.assertIn("Plots (2)", reader.pages[1].extract_text())

    def test_mixed_grid_and_figure_preserves_call_order(self):
        path = self._path("mixed.pdf")
        report = PdfChartReport(path, layout=(1, 2))
        report.new_page(suptitle="Grid First")
        report.get_next_ax().set_title("First Grid Plot")
        report.add_figure(self._line_figure("Middle Figure"), bookmark_title="Middle")
        report.new_page(layout=(1, 1), suptitle="Grid Last")
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

    def test_external_axes_format_copy_and_vector_output(self):
        path = self._path("vector.pdf")
        source_fig = self._line_figure("Source Plot")
        report = PdfChartReport(path, layout=(1, 1))
        report.add_external_axes([source_fig.axes[0]], copy_figure_format=True)
        report.save()
        plt.close(source_fig)

        reader = PdfReader(path)
        self.assertEqual(len(reader.pages), 1)
        self.assertEqual(reader.outline[0].title, "Page 1")
        self.assertIn("Source Plot", reader.pages[0].extract_text())
        xobjects = reader.pages[0]["/Resources"].get("/XObject", {})
        self.assertFalse(any(obj.get_object().get("/Subtype") == "/Image"
                             for obj in xobjects.values()))


if __name__ == "__main__":
    unittest.main()
