import unittest
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from quantbullet.reporting.pdf_text_report import PdfTextReport, PdfColumnFormat, PdfColumnMeta
from pathlib import Path

class TestPDFTextReport(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "./tests/_cache_dir"
        # just remove all files in the cache dir, but not the dir itself
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

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
        report.add_df_table( test_df, schema=schema )
        report.add_table_footnote("This is a footnote for the table.")
        report.add_page_break()

        test_fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3], [4, 5, 6])
        ax.set_title("Test Plot")

        report.add_matplotlib_figure(test_fig)
        report.save()