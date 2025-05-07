import unittest
import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
from quantbullet.model import FeatureScaledKNNRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

def make_mahalanobis_friendly_data(n_samples=500):
    """Generate synthetic data that is friendly for Mahalanobis distance.
    
    This data is generated to have strong correlations and non-linear relationships.
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, 3)
    # Add strong correlation
    X[:, 1] = X[:, 0] * 0.9 + np.random.randn(n_samples) * 0.1
    X[:, 2] = X[:, 0] * -0.8 + np.random.randn(n_samples) * 0.2
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n_samples)
    return X, y

def make_euclidean_friendly_data(n_samples=500):
    """Generate synthetic data that is friendly for Euclidean distance.
    
    Euclidean distance with few neighbors should work well with this data, as it is not strongly correlated and has a linear relationship.
    The prediction error would be small when observations are close in their feature space.
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, 3)
    y = 2 * X[:, 0] + 0.5 * X[:, 1] + 0.1 * X[:, 2] + 0.2 * np.random.randn(n_samples)
    return X, y

class TestFeatureScaledKNNRegressor(unittest.TestCase):
    def setUp(self):
        pass

    def test_naive(self):
        model = FeatureScaledKNNRegressor(n_neighbors=20, feature_weights=None)
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

    def test_check_estimator(self):
        # Check if the estimator passes sklearn's checks
        model = FeatureScaledKNNRegressor()
        check_estimator(model, generate_only=True)

    def test_grid_cv_euclidean(self):
        X, y = make_euclidean_friendly_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        # GridSearch
        param_grid = {
            'n_neighbors': [3, 5, 10],
            'metrics': ['euclidean', 'mahalanobis'],
            'weights': ['uniform', 'distance'],
            'feature_weights': [None]
        }
        grid = GridSearchCV(
            FeatureScaledKNNRegressor(),
            param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        self.assertEqual(getattr(best_model, 'metrics'), 'euclidean')

    def test_grid_cv_mahalanobis(self):
        X, y = make_mahalanobis_friendly_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        # GridSearch
        param_grid = {
            'n_neighbors': [3, 5, 10],
            'metrics': ['euclidean', 'mahalanobis'],
            'weights': ['uniform', 'distance'],
            'feature_weights': [None]
        }
        grid = GridSearchCV(
            FeatureScaledKNNRegressor(),
            param_grid,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        self.assertEqual(getattr(best_model, 'metrics'), 'mahalanobis')