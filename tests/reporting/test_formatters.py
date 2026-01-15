import unittest
import numpy as np
import pandas as pd

from quantbullet.reporting.formatters import (
    flex_number_formatter,
    number2string,
    human_number,
)


class TestFormatters(unittest.TestCase):
    def test_flex_number_formatter(self):
        self.assertEqual(flex_number_formatter(1234.567, decimals=1, comma=True), "1,234.6")
        self.assertEqual(flex_number_formatter(0.1234, percent=True), "12.34%")
        self.assertEqual(flex_number_formatter(5, decimals=0, transformer=lambda x: x * 2), "10")
        self.assertEqual(flex_number_formatter(np.nan), "")

    def test_number2string(self):
        self.assertEqual(number2string(999), "999")
        self.assertEqual(number2string(1234), "1,234")

        # float near int -> print as int
        self.assertEqual(number2string(1234.0000000001), "1,234")

        # "smart decimals" behavior
        self.assertEqual(number2string(12.3456), "12.35")   # 2 decimals are "good enough"
        self.assertEqual(number2string(0.123456), "0.1235") # 4 decimals needed to avoid distortion

        # scientific fallback
        self.assertEqual(number2string(1e-8), "1.00e-08")

    def test_human_number(self):
        self.assertEqual(human_number(None), "")
        self.assertEqual(human_number(np.nan), "")
        self.assertEqual(human_number(0), "0")

        # very small -> scientific (threshold default: 1e-2)
        self.assertEqual(human_number(0.0005), "5.00e-04")

        # < 1 uses more decimals (default decimals_lt1=4)
        self.assertEqual(human_number(0.1234), "0.1234")
        self.assertEqual(human_number(0.12), "0.12")  # trimmed

        # < 1000 uses fixed decimals (default decimals=2) + trimmed
        self.assertEqual(human_number(12.3), "12.3")

        # suffixes
        self.assertEqual(human_number(1200), "1.2K")
        self.assertEqual(human_number(1_234_567), "1.23M")
        self.assertEqual(human_number(-1500), "-1.5K")


