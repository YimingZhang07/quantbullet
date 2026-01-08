import unittest
import pandas as pd
import numpy as np
from datetime import date
from quantbullet.utils.cast import df_columns_to_tuples, to_jsonable, from_jsonable

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


    def test_to_jsonable_and_from_jsonable(self):
        test_dict = {
            'int': 1,
            'float': 1.5,
            'str': 'test',
            'bool': True,
            'none': None,
            'date': date(2023, 1, 1),
            'list': [1, 2, 3],
            'dict': {'a': 1, 'b': 2},
            'df': pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        }

        jsonable = to_jsonable(test_dict)
        reconstructed = from_jsonable(jsonable)
        self.assertEqual(reconstructed['int'], test_dict['int'])
        self.assertEqual(reconstructed['float'], test_dict['float'])
        self.assertEqual(reconstructed['str'], test_dict['str'])
        self.assertEqual(reconstructed['bool'], test_dict['bool'])
        self.assertEqual(reconstructed['none'], test_dict['none'])
        self.assertEqual(reconstructed['date'], '2023-01-01')