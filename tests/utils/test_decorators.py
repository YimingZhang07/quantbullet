import unittest
from datetime import date
from quantbullet.utils.decorators import normalize_date_args

@normalize_date_args("as_of_date", "settle_date")
def func(as_of_date, settle_date=None):
    return as_of_date, settle_date

class TestDecorators(unittest.TestCase):
    def test_normalize_date_args(self):
        # Test with date objects
        as_of_date = date(2023, 10, 1)
        settle_date = date(2023, 10, 2)
        result = func(as_of_date, settle_date)
        self.assertEqual(result, (date(2023, 10, 1), date(2023, 10, 2)))

        # Test with string dates
        as_of_date_str = "20231001"
        settle_date_str = "20231002"
        result = func(as_of_date_str, settle_date_str)
        self.assertEqual(result, (date(2023, 10, 1), date(2023, 10, 2)))

        # Test with string dates
        as_of_date_str = "2023-10-01"
        settle_date_str = "2023-10-02"
        result = func(as_of_date_str, settle_date_str)
        self.assertEqual(result, (date(2023, 10, 1), date(2023, 10, 2)))

        # Test with mixed types
        result = func(as_of_date_str, settle_date)
        self.assertEqual(result, (date(2023, 10, 1), date(2023, 10, 2)))