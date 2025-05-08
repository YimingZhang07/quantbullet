import numpy as np
import pandas as pd
import unittest
from copy import deepcopy
from sklearn.random_projection import sample_without_replacement
from sklearn.utils.estimator_checks import check_estimator

try:
    from quantbullet.model import ModelMetricsConsts
    from quantbullet.model import TimeWeightedXGBRegressor
    
except ImportError:
    ModelMetricsConsts = None
    TimeWeightedXGBRegressor = None

@unittest.skipIf(ModelMetricsConsts is None, "ModelMetricsConsts not available")
@unittest.skipIf(TimeWeightedXGBRegressor is None, "TimeWeightedXGBRegressor not available")
class TestModelSelection(object):
    def setup_class(self):
        pass

    def test_model_objective(self):
        assert isinstance(ModelMetricsConsts.xgboost_objectives, list)

    def test_time_weighted_xgboost(self):
        model = TimeWeightedXGBRegressor(
            reference_column="date", decay_rate=0.1, offset_days=0, alpha=0.1
        )
        # getting/setting params
        assert model.decay_rate == 0.1
        assert model.offset_days == 0

        params = model.get_params()
        assert params.get("decay_rate") == 0.1

        model.set_params(decay_rate=0.2)
        assert model.decay_rate == 0.2

        # fit/predict
        sample_data = {
            "date": pd.date_range(start="2020-01-01", end="2020-01-10"),
            "x": np.arange(10),
            "y": np.arange(10),
        }

        X = pd.DataFrame(sample_data)
        y = np.arange(10)
        X_before = deepcopy(X)

        model.fit(X, y)
        model.predict(X)

        # check if X is not modified
        assert X.equals(X_before)
