import unittest
import pandas as pd
from datetime import date
from quantbullet.reporting import ExcelExporter, HTMLPageBuilder, HTMLTableBuilder
from unittest.mock import patch, MagicMock

class TestExcelExporter(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "A": [1, 2],
            "B": [3, 4],
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"])
        })

    def test_set_default_decimals(self):
        exporter = ExcelExporter("dummy.xlsx")
        exporter.set_default_decimals(3)
        self.assertEqual(exporter.default_decimals, 3)

    def test_set_default_date_format(self):
        exporter = ExcelExporter("dummy.xlsx")
        exporter.set_default_date_format("yy-mm-dd")
        self.assertEqual(exporter.default_date_format, "yy-mm-dd")

    def test_set_overwrite(self):
        exporter = ExcelExporter("dummy.xlsx")
        exporter.set_overwrite(False)
        self.assertFalse(exporter.overwrite)

    def test_add_sheet(self):
        exporter = ExcelExporter("dummy.xlsx")
        exporter.add_sheet("TestSheet", self.df)
        self.assertEqual(len(exporter._sheets), 1)
        self.assertEqual(exporter._sheets[0]["sheet_name"], "TestSheet")

    @patch("pandas.DataFrame.to_excel")  # prevent real Excel writing
    @patch("pandas.ExcelWriter")         # prevent real ExcelWriter
    def test_save(self, mock_writer_cls, mock_to_excel):
        mock_writer = MagicMock()
        mock_writer_cls.return_value.__enter__.return_value = mock_writer
        mock_writer.sheets = {"Sheet1": MagicMock()}

        # use a chain of methods to call the ExcelExporter
        (
            ExcelExporter("dummy.xlsx")
            .add_sheet("Sheet1", self.df)
            .save()
        )

        # Assert writer was created correctly
        mock_writer_cls.assert_called_once_with("dummy.xlsx", engine="openpyxl", mode="w")

        # Assert DataFrame.to_excel was called
        mock_to_excel.assert_called_once_with(mock_writer, sheet_name="Sheet1", index=False)
        
class TestHTMLBuilders(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'Name'  : [ 'Alice', 'Bob' ],
            'Score' : [ 88.1234, 92.4567 ],
            'Status': [ 'pass', 'fail' ],
            'Date'  : [ date(2025, 1, 1), date(2025, 1, 2) ]
        })
        
        
        self.column_settings = {
            'Name'  : {'width': '10%', 'align': 'left'},
            'Score' : {'width': '10%', 'align': 'right', 'formatter': lambda x: f'{x:.1f}'},
            'Status': {'width': '10%', 'align': 'center', 'formatter': str.upper},
            'Date'  : {'width': '10%', 'align': 'center', 'formatter': lambda x: x.strftime('%Y-%m-%d')}
        }

    def test_build_html_page( self ):
        """A trivial test just to check if the HTML page builder works."""
        table_html = HTMLTableBuilder(
            self.df, 
            title="Test Results", 
            column_settings=self.column_settings,
            inline_style=True
        ).to_html()

        # Assemble page
        page = HTMLPageBuilder()
        page.add_heading("Quarterly Report", level=1)
        page.add_paragraph("The following tables summarize recent performance.")
        page.add_table(table_html)
        html_output = page.build()
        self.assertIn("<h1>Quarterly Report</h1>", html_output)
        # with open("preview.html", "w", encoding="utf-8") as f:
        #     f.write(html_output)

        # import webbrowser
        # webbrowser.open("preview.html")
