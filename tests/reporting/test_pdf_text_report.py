import unittest
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quantbullet.reporting.pdf_text_report import PdfTextReport, PdfColumnFormat, PdfColumnMeta, make_diverging_colormap
from pathlib import Path
from quantbullet.dfutils import sort_multiindex_by_hierarchy
from quantbullet.reporting.formatters import flex_number_formatter

class TestPDFTextReport(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "./tests/_cache_dir"
        # just remove all files in the cache dir, but not the dir itself
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        pass
    
    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        pass

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