import pandas as pd
import unittest
from quantbullet.dfutils import filter_df

class TestDFUtils(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'Name'  : ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'Age'   : [26, 27, 28, 29, 30],
            'Region': ['US', 'EU', 'US', 'EU', 'US'],
            'Score' : [85, 90, 95, 100, 100],
            'Status': ['active', 'inactive', 'active', 'active', 'inactive']
        })

    def test_filter_df(self):
        filters = [
            ("Age", ">=", 25),
            ("Name", "f", lambda s: s.str.startswith("A")),
            ("Region", "in", ["US", "EU"]),
            ("Score", "between", (80, 100)),
            ("Status", "!=", "inactive")
        ]

        # this should return only Alice
        filtered = filter_df(self.df, filters)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]['Name'], 'Alice')
        self.assertEqual(filtered.iloc[0]['Age'], 26)
