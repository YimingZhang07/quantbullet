import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypdf import PdfReader
from reportlab.platypus import Image

from quantbullet.plot.panels import PanelSet, panel_grid
from quantbullet.reporting.pdf_text_report import PdfTextReport, PdfColumnFormat, PdfColumnMeta, copy_panel, make_diverging_colormap, primary_axes
from pathlib import Path
from quantbullet.dfutils import sort_multiindex_by_hierarchy
from quantbullet.reporting.formatters import flex_number_formatter
from tests.artifacts import artifact_dir

class TestPDFTextReport(unittest.TestCase):
    def setUp(self):
        self.cache_dir = artifact_dir(self, "reporting/pdf_text_report")

    def test_pdf_text_report_main( self ):
        report = PdfTextReport( file_path=str( Path(self.cache_dir) / "test_report.pdf" ), report_title="Test Report", page_numbering=True )
        report.add_centered_text("This is a centered text.")
        test_df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4.5678, 5.6789, 6.7890],
            "C": ["foo", "bar", "baz"]
        })
        schema = [
            PdfColumnMeta(name="A", display_name="Column A", format=PdfColumnFormat(decimals=0)),
            PdfColumnMeta(name="B", display_name="Column B", format=PdfColumnFormat(decimals=2, comma=True, transformer=lambda x: x * 1000)),
            PdfColumnMeta(name="C", display_name="Column C", format=PdfColumnFormat())
        ]
        report.add_df_table( test_df, schema=schema, heatmap_all=True )
        report.add_table_footnote("This is a footnote for the table.")
        report.add_page_break()

        test_fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [4, 5, 6])
        ax.set_title("Test Plot")

        report.add_matplotlib_figure(test_fig)
        report.save()

    def test_pdf_text_report_multiindex_table( self ):
        # make a multiindex df
        # the columns are a multiindex with 2 levels
        arrays = [
            ['Group1', 'Group1', 'Group2', 'Group2'],
            ['Metric1', 'Metric2', 'Metric1', 'Metric2']
        ]

        # the row index is also a multiindex with 2 levels
        row_arrays = [
            ['A', 'A', 'B', 'B'],
            ['X', 'Y', 'X', 'Y']
        ]
        index = pd.MultiIndex.from_arrays(row_arrays, names=('Category', 'Subcategory'))
        columns = pd.MultiIndex.from_arrays(arrays, names=('Group', 'Metric'))
        data = [
            [13, 2, 3, 4],
            [2, 6, 7, 8],
            [7, 10, 11, 12],
            [1, 14, 15, 16]
        ]
        multiindex_df = pd.DataFrame(data, index=index, columns=columns)

        # sort the multiindex df by hierarchy
        row_order = {
            0: ['B', 'A'],  # Category level
            1: ['X', 'Y']   # Subcategory level
        }
        col_order = {
            'Group': ['Group2', 'Group1'],  # Group level
            'Metric': ['Metric2', 'Metric1'] # Metric level
        }
        multiindex_df = sort_multiindex_by_hierarchy(multiindex_df, row_orders=row_order, col_orders=col_order)

        # turn each column to some strings
        for col in multiindex_df.columns:
            multiindex_df[col] = multiindex_df[col].apply(lambda x: flex_number_formatter(x, decimals=0, comma=False))

        report = PdfTextReport( file_path=str( Path(self.cache_dir) / "test_report_multiindex.pdf" ),
                                report_title="Test Report MultiIndex", page_numbering=True )
        report.add_multiindex_df_table( multiindex_df, font_size=10, heatmap_all=True )
        report.save()

    def test_df_table_breakdown( self ):
        report = PdfTextReport( file_path=str( Path(self.cache_dir) / "test_report_breakdown.pdf" ) )

        test_df = pd.DataFrame({
            "A": np.arange(1, 41),
            "B": np.random.rand(40) * 100,
        })

        schema = [
            PdfColumnMeta(name="A", display_name="Column A", format=PdfColumnFormat(decimals=0, colormap=make_diverging_colormap(high_color="#63be7b", mid_color=(1, 1, 1), low_color="#f8696b", vmid=0))),
            PdfColumnMeta(name="B", display_name="Column B", format=PdfColumnFormat(decimals=2, comma=True, headerbgcolor="#63be7b") ),
        ]

        report.add_df_table_breakdown( test_df, schema=schema, nrows=10 )
        
        report.add_page_break()
        report.add_df_table( test_df, schema=schema )

        report.save()

    def test_convenience_methods(self):
        pdf_path = str(Path(self.cache_dir) / "test_convenience.pdf")
        report = PdfTextReport(file_path=pdf_path, report_title="Convenience API Test")

        # add_heading — all three levels
        report.add_heading("Level 1 Heading")
        report.add_heading("Level 2 Heading", level=2)
        report.add_heading("Special chars: A & B < C > D", level=3)
        report.add_heading("No bookmark heading", level=2, bookmark=False)

        # add_body
        report.add_body("This is a body paragraph with <b>bold</b> and <i>italic</i> text.")
        report.add_body("Another paragraph.", font_size=10, space_after=12)

        # add_kv_line — single pair
        report.add_kv_line("MAE", "31.2 bps")

        # add_kv_line — multiple pairs via dict
        report.add_kv_line({"MAE": "31.2 bps", "RMSE": "45.1 bps", "R2": "0.55"})

        # add_kv_line — custom separator
        report.add_kv_line({"Alpha": "0.05", "Beta": "0.95"}, separator=" / ")

        # add_list — unordered
        report.add_list([
            "First bullet item",
            "Second with <b>bold</b>",
            "Third item",
        ])

        # add_list — ordered
        report.add_list([
            "<b>Signal:</b> Pearson r = 0.124",
            "<b>Partial corr:</b> r = 0.085",
            "<b>IV:</b> 0.0312",
        ], ordered=True)

        report.add_page_break()
        report.add_heading("Page 2: Mixed Content", level=1)
        report.add_body("Body text after a heading on page 2.")
        report.add_kv_line("Metric", "42.0")
        report.add_list(["Item A", "Item B"], ordered=False, font_size=10)

        report.save()
        self.assertTrue(Path(pdf_path).exists())
        self.assertGreater(Path(pdf_path).stat().st_size, 0)

    def test_paged_figure_keeps_count_bars_on_the_curve_axis(self):
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        for ax in axes.flat:
            ax.plot([0, 1], [1, 2], label="curve")
            twin = ax.twinx()
            twin.bar([0, 1], [3, 4], width=0.2)
        self.assertEqual(len(fig.axes), 8)
        self.assertEqual(len(primary_axes(fig)), 4)
        _, dst_ax = plt.subplots()
        copy_panel(axes.flat[0], dst_ax)
        siblings = [ax for ax in dst_ax._twinned_axes.get_siblings(dst_ax) if ax is not dst_ax]
        self.assertEqual(len(dst_ax.get_lines()), 1)
        self.assertEqual(len(siblings), 1)
        self.assertEqual(len(siblings[0].patches), 2)
        plt.close("all")

    def test_print_sized_matrix_stays_one_image_and_tall_matrix_splits_by_row(self):
        report = PdfTextReport(file_path=str(Path(self.cache_dir) / "paged.pdf"), page_numbering=False)
        content_w, _ = report.get_page_dimensions()
        fig, axes = plt.subplots(3, 3, figsize=(10, 6))
        for ax in axes.flat:
            ax.plot([0, 1], [0, 1])
            ax.twinx().bar([0, 1], [1, 2], width=0.2)
        report.add_matplotlib_figure_paged(fig, n_cols=3, dpi=40)
        images = [item for item in report.story if isinstance(item, Image)]
        self.assertEqual(len(images), 1)
        self.assertLessEqual(images[0].drawWidth, content_w * 1.01)
        self.assertGreater(images[0].drawWidth, content_w * 0.7)

        tall, tall_axes = plt.subplots(6, 3, figsize=(10, 20))
        for ax in tall_axes.flat:
            ax.plot([0, 1], [0, 1])
        before = len(images)
        report.add_matplotlib_figure_paged(tall, n_cols=3, dpi=30)
        pages = [item for item in report.story if isinstance(item, Image)]
        self.assertGreater(len(pages), before + 1)
        for image in pages[before:]:
            self.assertLessEqual(image.drawWidth, content_w * 1.01)
            self.assertGreater(image.drawWidth, content_w * 0.7)
        plt.close("all")


def _line(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot([0, 1], [0, 1])
    return fig


def _grid_recorder(calls):
    def render(panels, n_cols, panel_size):
        calls.append((panels, panel_size))
        return panel_grid(len(panels), n_cols, panel_size)[0]
    return render


class TestSlotSizedFigures(unittest.TestCase):
    """Figures are drawn at the size the page gives them and placed 1:1."""

    def setUp(self):
        self.cache_dir = artifact_dir(self, "reporting/pdf_text_report")

    def tearDown(self):
        plt.close("all")

    def report(self, name):
        path = Path(self.cache_dir) / name
        return path, PdfTextReport(file_path=str(path), page_numbering=False)

    @staticmethod
    def pages(path):
        reader = PdfReader(str(path))
        return [len(page.images) for page in reader.pages], [page.extract_text() for page in reader.pages]

    def test_fixed_and_fill_figures_get_the_frame_width(self):
        path, report = self.report("slot_figures.pdf")
        frame_w, frame_h = report.content_size_inches()
        sizes = []
        def draw(width, height):
            sizes.append((width, height))
            return _line(width, height)
        report.add_heading("Fixed")
        report.add_figure(draw, height=3)
        report.add_heading("Fill")
        report.add_figure(draw)
        report.save()
        self.assertEqual(len(sizes), 2)
        self.assertAlmostEqual(sizes[0][0], frame_w)
        self.assertAlmostEqual(sizes[0][1], 3)
        self.assertAlmostEqual(sizes[1][0], frame_w)
        self.assertTrue(3 < sizes[1][1] < frame_h - 3)  # the rest of page 1
        self.assertEqual(self.pages(path)[0], [2])

    def test_draw_must_return_the_slot_size(self):
        _, report = self.report("wrong_size.pdf")
        report.add_figure(lambda width, height: _line(width / 2, height))
        with self.assertRaisesRegex(ValueError, "slot"):
            report.save()

    def test_heading_moves_with_a_figure_that_does_not_fit(self):
        path, report = self.report("keep_heading.pdf")
        _, frame_h = report.content_size_inches()
        report.add_figure(_line, height=frame_h - 1)
        report.add_heading("Next chart")
        report.add_figure(_line, height=2)
        report.save()
        images, texts = self.pages(path)
        self.assertEqual(images, [1, 1])
        self.assertNotIn("Next chart", texts[0])
        self.assertIn("Next chart", texts[1])

    def test_grid_that_fits_a_fresh_page_moves_there_with_its_heading(self):
        path, report = self.report("grid_moves.pdf")
        calls = []
        report.add_figure(_line, height=4)
        report.add_heading("Grid")
        report.add_figure_grid(PanelSet([f"p{i}" for i in range(6)], _grid_recorder(calls), n_cols=3))
        report.save()
        images, texts = self.pages(path)
        self.assertEqual(len(calls), 1)
        self.assertEqual(images, [1, 1])
        self.assertIn("Grid", texts[1])

    def test_tall_grid_splits_by_rows_at_one_panel_size(self):
        path, report = self.report("grid_split.pdf")
        frame_w, _ = report.content_size_inches()
        calls = []
        names = [f"p{i}" for i in range(15)]
        report.add_heading("Grid")
        report.add_figure_grid(PanelSet(names, _grid_recorder(calls), n_cols=3), min_panel_height=2)
        report.save()
        self.assertEqual([panels for panels, _ in calls], [tuple(names[:9]), tuple(names[9:])])
        (first_w, first_h), (rest_w, rest_h) = (size for _, size in calls)
        self.assertAlmostEqual(first_w, frame_w / 3)
        self.assertAlmostEqual(rest_w, first_w)
        self.assertAlmostEqual(rest_h, first_h)
        self.assertGreaterEqual(first_h, 2)
        images, texts = self.pages(path)
        self.assertEqual(images, [1, 1])
        self.assertIn("Grid (continued)", texts[1])
