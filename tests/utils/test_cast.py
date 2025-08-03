import unittest
import pandas as pd
import numpy as np
from datetime import date
from quantbullet.utils.cast import df_columns_to_tuples

class TestCastUtils(unittest.TestCase):
    def test_df_columns_to_tuples(self):
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'value': [10, 20, 30],
            'string': ['a', 'b', 'c']
        })
        df['date'] = pd.to_datetime(df['date'])
        result = df_columns_to_tuples(df['date'], df['value'], df['string'])
        expected = [
            (date(2023, 1, 1), 10, 'a'),
            (date(2023, 1, 2), 20, 'b'),
            (date(2023, 1, 3), 30, 'c')
        ]
        self.assertEqual(result, expected)