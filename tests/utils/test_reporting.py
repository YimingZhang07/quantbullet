import unittest
import pandas as pd
from quantbullet.utils.reporting import ExcelExporter
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
        exporter.add_sheet(self.df, "TestSheet")
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
            .add_sheet(self.df, "Sheet1")
            .save()
        )

        # Assert writer was created correctly
        mock_writer_cls.assert_called_once_with("dummy.xlsx", engine="openpyxl", mode="w")

        # Assert DataFrame.to_excel was called
        mock_to_excel.assert_called_once_with(mock_writer, sheet_name="Sheet1", index=False)