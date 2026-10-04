import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from sklearn.preprocessing import OneHotEncoder

from quantbullet.linear_product_model import (
    LinearProductRegressorBCD,
    LinearProductModelToolkit,
)
from quantbullet.preprocessing import FlatRampTransformer
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
from quantbullet.linear_product_model.base import InteractionCoef
from quantbullet.model.feature import DataType, Feature, FeatureRole, FeatureSpec
from tests.artifacts import artifact_dir


class _FrozenAgeCurve:
    def predict(self, basis):
        return 1. + basis[:, 1]


def _scale_transfer_case(loss, weights, *, frozen_curve=False, frozen_by=False, dropped_by=False):
    """Two independently fitted curves, with category order opposite column order."""
    model = LinearProductRegressorBCD()
    model.loss_ = loss
    model.interactions_ = {'age': 'purpose'}
    model.global_scalar_ = 1.0
    model.submodels_ = {'purpose': object()} if frozen_by else {}
    basis = np.array([[1., 1.], [1., 2.], [1., 1.], [1., 2.]])
    by = np.array([[0., 1.], [0., 1.], [1., 0.], [1., 0.]])
    by_coef = np.array([1.2, .8])
    if dropped_by:
        by = by[:, :1]
        by_coef = by_coef[:1]
    data_blocks = {'age': basis, 'purpose': by}
    masks = {'age': {'P': np.array([True, True, False, False]),
                     'R': np.array([False, False, True, True])}}
    curves = {'P': np.array([1., 0.]), 'R': np.array([1., 0.])}
    if frozen_curve:
        curves['R'] = _FrozenAgeCurve()
    interaction_params = {'age': curves}
    params_blocks = {'purpose': by_coef}
    block_preds = {
        'age': model._build_interaction_block_pred('age', basis, interaction_params, masks),
        'purpose': by @ by_coef,
        'other': np.array([.001, .1, .02, .03]),
    }
    y = np.array([.05, .051, .014, .037])
    return model, data_blocks, block_preds, interaction_params, masks, params_blocks, y


def _case_prediction(case):
    model, _, blocks, *_ = case
    return model.global_scalar_ * blocks['age'] * blocks['purpose'] * blocks['other']


@pytest.mark.parametrize('loss', ['mse', 'poisson'])
@pytest.mark.parametrize('weights', [None, np.array([1., 3., 2., 5.])])
@pytest.mark.parametrize('frozen_curve', [False, True])
def test_interaction_normalization_preserves_candidate_predictions(loss, weights, frozen_curve):
    raw = _scale_transfer_case(loss, weights, frozen_curve=frozen_curve)
    normalized = _scale_transfer_case(loss, weights, frozen_curve=frozen_curve)
    initial_prediction = _case_prediction(normalized)

    model, data, blocks, curves, masks, params, y = raw
    # No level transfer means the fitted curves deliberately remain unnormalized.
    model._step_interaction_group('age', data, blocks, curves, masks, y, weights)
    expected = _case_prediction(raw)
    raw_means = {cat: np.average(blocks['age'][mask],
                                weights=None if weights is None else weights[mask])
                 for cat, mask in masks['age'].items() if isinstance(curves['age'][cat], np.ndarray)}

    model, data, blocks, curves, masks, params, y = normalized
    columns = model._interaction_level_columns('age', data, masks, params)
    assert columns == {'P': 1, 'R': 0}
    model._step_interaction_group(
        'age', data, blocks, curves, masks, y, weights,
        params_blocks=params, interaction_level_columns={'age': columns})
    actual = _case_prediction(normalized)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    assert model.loss_function(actual, y, weights) == pytest.approx(
        raw[0].loss_function(expected, y, weights), abs=1e-14)
    assert model.loss_function(actual, y, weights) < model.loss_function(initial_prediction, y, weights)

    for cat, mask in masks['age'].items():
        if isinstance(curves['age'][cat], np.ndarray):
            assert np.average(blocks['age'][mask],
                              weights=None if weights is None else weights[mask]) == pytest.approx(1.)
    assert np.average(blocks['purpose'], weights=weights) == pytest.approx(1.)
    expected_level_ratio = (1.2 * raw_means.get('R', 1.)) / (.8 * raw_means['P'])
    assert params['purpose'][0] / params['purpose'][1] == pytest.approx(expected_level_ratio)
    np.testing.assert_allclose(blocks['purpose'], data['purpose'] @ params['purpose'])
    np.testing.assert_allclose(blocks['age'], model._build_interaction_block_pred(
        'age', data['age'], curves, masks))
    if frozen_curve:
        np.testing.assert_allclose(blocks['age'][masks['age']['R']], [2., 3.])


@pytest.mark.parametrize('frozen_by,dropped_by', [(True, False), (False, True)])
def test_interaction_keeps_curve_scale_when_by_block_cannot_absorb_it(frozen_by, dropped_by):
    case = _scale_transfer_case('mse', None, frozen_by=frozen_by, dropped_by=dropped_by)
    model, data, blocks, curves, masks, params, y = case
    old_coef = params['purpose'].copy()
    old_by_pred = blocks['purpose'].copy()
    columns = model._interaction_level_columns('age', data, masks, params)
    assert columns is None
    model._step_interaction_group(
        'age', data, blocks, curves, masks, y, None,
        params_blocks=params, interaction_level_columns={'age': columns})
    np.testing.assert_array_equal(params['purpose'], old_coef)
    np.testing.assert_array_equal(blocks['purpose'], old_by_pred)
    assert model.global_scalar_ == 1.
    assert not np.isclose(blocks['age'][masks['age']['P']].mean(), 1.)


def _generate_interaction_data(n_samples=50_000, seed=42):
    """Generate synthetic data: y = C(x2) * x1^2 + noise + intercept(x2).

    x2 is categorical with two levels: 'A' (30%) and 'B' (70%).
    Both categories share the same quadratic shape in x1, but category B
    has a steeper slope and higher intercept.

    Returns (df, weights) where weights are random positive per-observation weights.
    """
    np.random.seed(seed)

    x2 = np.random.choice(['A', 'B'], size=n_samples, p=[0.3, 0.7])
    x1 = 3 * np.random.randn(n_samples)

    f_x1 = x1 ** 2

    y = np.where(
        x2 == 'A',
        1.0 * f_x1 + np.random.randn(n_samples) * 5.0 + 1,
        5.0 * f_x1 + np.random.randn(n_samples) * 5.0 + 5,
    )

    weights = np.random.exponential(scale=1.0, size=n_samples)

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    df['x2'] = df['x2'].astype('category')
    return df, weights


class TestInteraction(unittest.TestCase):
    def setUp(self):
        self.cache_dir = artifact_dir(self, "linear_product_model/interaction")

    def test_interaction_x1_by_x2(self):
        df, weights = _generate_interaction_data()

        preprocess_config = {
            'x1': FlatRampTransformer(
                knots=list(np.arange(-9, 10, 1)),
                include_bias=True,
            ),
            'x2': OneHotEncoder(),
        }

        feature_spec = FeatureSpec(features=[
            Feature(name='x1', dtype=DataType.FLOAT, role=FeatureRole.MODEL_INPUT),
            Feature(name='x2', dtype=DataType.CATEGORY, role=FeatureRole.MODEL_INPUT),
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
        self.assertTrue(mse < 30, f"MSE too high: {mse:.4f}")

        # --- test that each block mean should be very close to 1 ---
        for key in tk.feature_groups:
            block_mean = model.block_means_[key]
            self.assertTrue(np.isclose(block_mean, 1, atol=1e-4), f"Block mean for '{key}' should be close to 1: {block_mean:.4f}")

        # --- implied-actual plots with sample_weights ---
        fig, axes = tk.plot_implied_actuals(model, dcontainer, sample_weights=weights)
        fig_path = Path(self.cache_dir) / "interaction_implied_actuals.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        self.assertTrue(fig_path.exists())
        plt.close(fig)


    def test_interaction_poisson_loss(self):
        df, weights = _generate_interaction_data()
        df['y'] = np.maximum(df['y'], 0.01)

        preprocess_config = {
            'x1': FlatRampTransformer(
                knots=list(np.arange(-9, 10, 1)),
                include_bias=True,
            ),
            'x2': OneHotEncoder(),
        }

        feature_spec = FeatureSpec(features=[
            Feature(name='x1', dtype=DataType.FLOAT, role=FeatureRole.MODEL_INPUT),
            Feature(name='x2', dtype=DataType.CATEGORY, role=FeatureRole.MODEL_INPUT),
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
            loss='poisson',
        )

        preds = model.predict(dcontainer)
        mse = np.mean((df['y'].values - preds) ** 2)
        print(f"Poisson interaction test MSE: {mse:.4f}")
        self.assertTrue(mse < 30, f"MSE too high: {mse:.4f}")

        self.assertIsInstance(model.coef_['x1'], InteractionCoef)
        self.assertEqual(model.loss_, 'poisson')

        # --- Poisson unbiasedness: weighted mean of fitted ≈ weighted mean of actual ---
        y_true = df['y'].values
        wmean_actual = np.average(y_true, weights=weights)
        wmean_fitted = np.average(preds, weights=weights)
        rel_err_overall = abs(wmean_fitted - wmean_actual) / wmean_actual
        print(f"  Overall  weighted mean: actual={wmean_actual:.4f}, fitted={wmean_fitted:.4f}, rel_err={rel_err_overall:.6f}")
        self.assertLess(rel_err_overall, 1e-4,
                        f"Overall weighted mean bias too large: {rel_err_overall:.8f}")

        for cat in sorted(df['x2'].unique()):
            mask = (df['x2'] == cat).values
            w_cat = weights[mask]
            wm_actual = np.average(y_true[mask], weights=w_cat)
            wm_fitted = np.average(preds[mask], weights=w_cat)
            rel_err = abs(wm_fitted - wm_actual) / wm_actual
            print(f"  Cat '{cat}' weighted mean: actual={wm_actual:.4f}, fitted={wm_fitted:.4f}, rel_err={rel_err:.8f}")
            self.assertLess(rel_err, 1e-4,
                            f"Weighted mean bias for category '{cat}' too large: {rel_err:.8f}")


if __name__ == '__main__':
    unittest.main()
