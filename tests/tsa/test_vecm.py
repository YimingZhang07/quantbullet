import unittest

import pandas as pd
import numpy as np

from quantbullet.tsa import VectorErrorCorrectionModel

class TestVECM(unittest.TestCase):
    def test_vecm_one_cointegration_factor(self):
        np.random.seed(0)
        date_range = pd.date_range(start="1/1/2020", periods=10, freq="D")
        data = pd.DataFrame(
            {
                "A": np.random.normal(0, 1, 10),
                "B": np.random.normal(0, 1, 10),
                "C": np.random.normal(0, 1, 10),
            },
            index=date_range,
        )
        vecm = VectorErrorCorrectionModel()
        vecm.fit(data)
        self.assertEqual(vecm.Beta.shape, (3, 1))
        self.assertEqual(vecm.LoadingMatrix.shape, (3, 1))
        self.assertEqual(vecm.getIntegratedSeries(data).shape, (10, 1))