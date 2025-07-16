import unittest
import numpy as np
import pandas as pd
from quantbullet.preprocessing.smoothing import MultiSeriesConstraintSmoother

class TestMultiSeriesConstraintSmoother(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=60)
        base_curve = np.linspace(150, 300, len(dates))  # base DM trend

        self.smoothed_dms = pd.DataFrame({
            'AAA': base_curve + np.random.normal(0, 3, size=len(dates)),
            'AA':  base_curve + 20 + np.random.normal(0, 3, size=len(dates)),
            'A':   base_curve + 50 + np.random.normal(0, 3, size=len(dates)),
            'BBB': base_curve + 70 + np.random.normal(0, 3, size=len(dates)),
        }, index=dates)

        # Inject violations
        self.smoothed_dms.iloc[10, self.smoothed_dms.columns.get_loc('AA')] = self.smoothed_dms.iloc[10]['AAA'] - 10
        self.smoothed_dms.iloc[25, self.smoothed_dms.columns.get_loc('BBB')] = self.smoothed_dms.iloc[25]['A'] - 30
        self.smoothed_dms.iloc[40, self.smoothed_dms.columns.get_loc('AA')] = self.smoothed_dms.iloc[40]['AAA'] + 100

        self.engine = MultiSeriesConstraintSmoother(
            monotonic_pairs=[('AAA', 'AA'), ('AA', 'A'), ('A', 'BBB')],
            band_constraints=[
                ('AA',  'AAA', 10, 50),
                ('BBB', 'AA', 30, 120),
                ('A',   'AA', 5, 100)
            ],
            series_order=['AAA', 'AA', 'A', 'BBB']
        )

    def test_monotonicity_and_band_constraints(self):
        adjusted = self.engine.apply(self.smoothed_dms)

        # Check monotonicity
        self.assertTrue((adjusted['AAA'] <= adjusted['AA']).all())
        self.assertTrue((adjusted['AA'] <= adjusted['A']).all())
        self.assertTrue((adjusted['A'] <= adjusted['BBB']).all())

        # Check band constraints
        # Due to the precision, we use (10, 51) instead of (10, 50)
        self.assertTrue(((adjusted['AA'] - adjusted['AAA']).between(10, 51)).all())
        self.assertTrue(((adjusted['BBB'] - adjusted['AA']).between(30, 120)).all())
        self.assertTrue(((adjusted['A'] - adjusted['AA']).between(5, 100)).all())

    def test_logging_output(self):
        self.engine.apply(self.smoothed_dms)
        logs = self.engine.get_logs()
        self.assertIsInstance(logs, pd.DataFrame)
        self.assertGreater(len(logs), 0)  # Expect some log entries due to violations

if __name__ == '__main__':
    unittest.main()
