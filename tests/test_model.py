import numpy as np
import pandas as pd
import unittest
from copy import deepcopy
from sklearn.random_projection import sample_without_replacement
from sklearn.utils.estimator_checks import check_estimator

try:
    from quantbullet.model import ModelMetricsConsts
    from quantbullet.model import TimeWeightedXGBRegressor
    from quantbullet.model import weightedDistanceKNRegressor
except ImportError:
    ModelMetricsConsts = None
    TimeWeightedXGBRegressor = None
    weightedDistanceKNRegressor = None

@unittest.skipIf(ModelMetricsConsts is None, "ModelMetricsConsts not available")
@unittest.skipIf(TimeWeightedXGBRegressor is None, "TimeWeightedXGBRegressor not available")
@unittest.skipIf(weightedDistanceKNRegressor is None, "weightedDistanceKNRegressor not available")
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

    def test_weighted_distance_kn_regressor(self):
        model = weightedDistanceKNRegressor(n_neighbors=20, feature_weights=None)

        # getting/setting params
        params = model.get_params()
        assert params.get("n_neighbors") == 20

        model.set_params(n_neighbors=30)
        assert model.n_neighbors == 30

        # BUG check estimator only works for a new instance of the model
        # check_estimator(model) fails
        # check_estimator(weightedDistanceKNRegressor(n_neighbors=20, feature_weights=None)) works

        model.set_params(n_neighbors=3)
        sample_train_data = {
            "X": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
        sample_test_X = [0, 1, 2]
        model.fit(pd.DataFrame(sample_train_data["X"]), sample_train_data["y"])
        res = model.predict(pd.DataFrame(sample_test_X))
        np.testing.assert_array_equal(res, np.array([2, 5, 8]))

        # test feature weights
        model.set_params(n_neighbors=2)
        model.set_params(feature_weights=[0, 1])
        sample_train_data = {
            "X": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "y": [1, 2, 3, 4],
        }
        # with feature weights, the first feature is ignored
        # and the n_neighbors is set to be 2. So the first and third test samples are essentially the same
        sample_test_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        model.fit(pd.DataFrame(sample_train_data["X"]), sample_train_data["y"])
        res = model.predict(pd.DataFrame(sample_test_X))
        np.testing.assert_array_equal(res, np.array([2, 3, 2, 3]))
