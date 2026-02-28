import unittest
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from quantbullet.linear_product_model import (
    LinearProductRegressorBCD,
    LinearProductModelToolkit,
)
from quantbullet.preprocessing import FlatRampTransformer
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
from quantbullet.linear_product_model.base import InteractionCoef
from quantbullet.model.feature import DataType, Feature, FeatureRole, FeatureSpec

DEV_MODE = True


def _generate_interaction_data(n_samples=50_000, seed=42):
    """Generate synthetic multiplicative data: y = C * f(x1 | x2) * g(x3) + noise.

    x2 is categorical with two levels: 'A' (30%) and 'B' (70%).
    Category A follows a U-shape in x1; category B follows a linear trend.

    Returns (df, weights) where weights are random positive per-observation weights.
    """
    np.random.seed(seed)

    x1 = 10 + 3 * np.random.randn(n_samples)
    x2 = np.random.choice(['A', 'B'], size=n_samples, p=[0.3, 0.7])
    x3 = 5 + 2 * np.random.randn(n_samples)

    f_x1 = np.where(
        x2 == 'A',
        1.0 + 0.03 * (x1 - 10) ** 2,   # U-shape (concave up)
        0.5 + 0.08 * x1,                 # linear increasing
    )
    g_x3 = 1.0 + 0.1 * np.cos(x3 * 0.5)

    y = 5.0 * f_x1 * g_x3 + np.random.randn(n_samples) * 2.0

    weights = np.random.exponential(scale=1.0, size=n_samples)
    weights[x2 == 'B'] *= 3.0

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y})
    df['x2'] = df['x2'].astype('category')
    return df, weights


class TestInteraction(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "./tests/_cache_dir"
        if DEV_MODE:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        else:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if not DEV_MODE:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_interaction_x1_by_x2(self):
        df, weights = _generate_interaction_data()

        preprocess_config = {
            'x1': FlatRampTransformer(
                knots=list(np.arange(4, 17, 1)),
                include_bias=True,
            ),
            'x2': OneHotEncoder(),
            'x3': FlatRampTransformer(
                knots=list(np.arange(1, 10, 1)),
                include_bias=True,
            ),
        }

        feature_spec = FeatureSpec(features=[
            Feature(name='x1', dtype=DataType.FLOAT, role=FeatureRole.MODEL_INPUT),
            Feature(name='x2', dtype=DataType.CATEGORY, role=FeatureRole.MODEL_INPUT),
            Feature(name='x3', dtype=DataType.FLOAT, role=FeatureRole.MODEL_INPUT),
            Feature(name='y', dtype=DataType.FLOAT, role=FeatureRole.TARGET),
        ])

        tk = LinearProductModelToolkit(
            feature_spec=feature_spec,
            preprocess_config=preprocess_config,
        ).fit(df)
        expanded_df = tk.get_expanded_df(df)

        dcontainer = ProductModelDataContainer(
            df, expanded_df, response=df['y'], feature_groups=tk.feature_groups,
        )

        model = LinearProductRegressorBCD()
        model.fit(
            dcontainer,
            feature_groups=tk.feature_groups,
            interactions={'x1': 'x2'},
            n_iterations=10,
            early_stopping_rounds=5,
            weights=weights,
        )

        # --- basic convergence checks ---
        preds = model.predict(dcontainer)
        mse = np.mean((df['y'].values - preds) ** 2)
        print(f"Interaction test MSE: {mse:.4f}")
        self.assertTrue(mse < 10, f"MSE too high: {mse:.4f}")

        # x1 should be stored as an InteractionCoef with two categories
        self.assertIsInstance(model.coef_['x1'], InteractionCoef)
        self.assertEqual(model.coef_['x1'].by, 'x2')
        self.assertEqual(set(model.coef_['x1'].categories.keys()), {'A', 'B'})

        # x3 should be a regular ndarray coefficient
        self.assertIsInstance(model.coef_['x3'], np.ndarray)

        # --- history tracking: interaction_params_history_ aligns with loss_history_ ---
        self.assertEqual(
            len(model.interaction_params_history_),
            len(model.loss_history_),
            "interaction_params_history_ length should match loss_history_",
        )
        self.assertIsNotNone(model.best_interaction_params_)

        # --- implied-actual plots with sample_weights ---
        fig, axes = tk.plot_implied_actuals(model, dcontainer, sample_weights=weights)
        fig_path = Path(self.cache_dir) / "interaction_implied_actuals.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        self.assertTrue(fig_path.exists())
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
