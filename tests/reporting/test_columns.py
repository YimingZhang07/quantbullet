import unittest
import pandas as pd
from quantbullet.reporting.columns import ColumnFormat, ColumnMeta
from quantbullet.reporting.pdf_text_report import PdfColumnMeta


class TestColumnMeta(unittest.TestCase):
    def test_display_name_defaults_to_name(self):
        self.assertEqual(ColumnMeta(name="a").display_name, "a")
        self.assertEqual(PdfColumnMeta(name="a").display_name, "a")

    def test_default_format_is_column_format(self):
        self.assertIsInstance(ColumnMeta(name="a").format, ColumnFormat)


class TestColumnFormat(unittest.TestCase):
    def test_repr_and_equality_include_excel_fields(self):
        self.assertIn("width=12", repr(ColumnFormat(width=12)))
        self.assertNotEqual(ColumnFormat(align="left"), ColumnFormat(align="right"))

    def test_date_format_none_means_default(self):
        self.assertEqual(ColumnFormat(date_format=None).date_format, "yyyy-mm-dd")

    def test_unknown_align(self):
        with self.assertRaises(ValueError):
            ColumnFormat(align="middle")


class TestEstimateDisplayWidth(unittest.TestCase):
    def test_numbers_as_excel_shows_them(self):
        self.assertEqual(ColumnFormat(decimals=2, percent=True).estimate_display_width(pd.Series([0.5]), ""),
                         len("50.00%"))
        self.assertEqual(ColumnFormat(decimals=0, comma=True, parens_for_negative=True)
                         .estimate_display_width(pd.Series([-1234.0]), ""), len("(1,234)"))

    def test_series_is_not_transformed_again(self):
        fmt = ColumnFormat(transformer=lambda x: x * 1e6)
        written = fmt.apply_transform(pd.Series([1.0, 2.0]))
        self.assertEqual(fmt.estimate_display_width(written, "v"), len("2000000.00"))

    def test_no_decimals_uses_general_format(self):
        fmt = ColumnFormat(decimals=None)
        self.assertEqual(fmt.estimate_display_width(pd.Series([1.23456]), "v"), len("1.23456"))
        self.assertEqual(fmt.estimate_display_width(pd.Series([-0.7443954656345405]), "v"), len("-0.7443954656"))

    def test_dates_use_date_format(self):
        dates = pd.Series([pd.Timestamp("2023-05-01"), pd.Timestamp("2023-09-05 14:07")])
        self.assertEqual(ColumnFormat(date_format="mmm yyyy").estimate_display_width(dates, "d"), len("Sep 2023"))
        self.assertEqual(ColumnFormat(date_format="mmmm d, yyyy").estimate_display_width(dates, "d"),
                         len("September 5, 2023"))
        self.assertEqual(ColumnFormat(date_format="yyyy-mm-dd h:mm AM/PM").estimate_display_width(dates, "d"),
                         len("2023-05-01 12:00 AM"))   # midnight is wider than 2:07 PM

    def test_empty_series_uses_header(self):
        self.assertEqual(ColumnFormat().estimate_display_width(pd.Series([], dtype=float), "header"), 6)


if __name__ == "__main__":
    unittest.main()
