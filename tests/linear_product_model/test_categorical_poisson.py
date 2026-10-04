"""Equivalence checks for grouped one-hot IRLS and its dense reference path."""

import pickle

import numpy as np
import pandas as pd
import pytest

from quantbullet.linear_product_model import LinearProductRegressorBCD
from quantbullet.linear_product_model._acceleration import one_hot_codes
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer


class DenseReference(LinearProductRegressorBCD):
    def _init_one_hot_codes(self, data_blocks, params_blocks):
        return {}


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_one_hot_codes_use_columns_and_allow_unused_categories(dtype):
    X = np.eye(5, dtype=dtype)[[3, 0, 2, 3, 1, 0, 1]]
    codes = one_hot_codes(X, chunk_size=2)
    np.testing.assert_array_equal(codes, [3, 0, 2, 3, 1, 0, 1])
    assert codes.dtype == np.int32


@pytest.mark.parametrize('X', [
    [[0., 0.], [0., 1.]],  # dropped category
    [[1., 1.], [0., 1.]],  # multi-hot
    [[.5, 0.], [0., 1.]],
    [[-1., 0.], [0., 1.]],
    [[np.nan, 0.], [0., 1.]],
    [[np.inf, 0.], [0., 1.]],
    [[1., 2.], [1., 3.]],  # continuous basis
])
def test_non_one_hot_bases_keep_dense_solver(X):
    assert one_hot_codes(np.asarray(X)) is None


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('use_floor', [False, True])
def test_grouped_poisson_matches_dense_irls_with_ridge(dtype, weighted, use_floor):
    codes = np.array([3, 0, 3, 0, 2, 1, 1, 0, 1], dtype=np.int32)
    X = np.eye(5, dtype=dtype)[codes]  # category 2 is rare; category 4 is unobserved
    y = np.array([.01, .02, .03, 0., .04, 0., 0., .01, 0.])
    fixed = np.linspace(.5, 1.5, len(codes))
    coef = np.array([.6, 1., 1.3, 1.4, .2])
    if use_floor:
        fixed[:2] = [-.2, 0.]
        coef[2:4] = [0., -.3]
    weights = np.array([1., 3., .5, 0., 2., 4., 1., .8, 2.]) if weighted else None
    model = LinearProductRegressorBCD()
    model.global_scalar_ = .01
    dense = model._solve_block_poisson(X, fixed, y, weights, coef)
    grouped = model._solve_one_hot_poisson(codes, fixed, y, weights, coef)
    np.testing.assert_allclose(grouped, dense, rtol=1e-10, atol=1e-12)
    assert grouped[1] == 0.  # observed category with no events
    assert grouped[4] == 0.


def make_fit_data(dtype=np.float64, n=3000):
    rng = np.random.default_rng(42)
    age = rng.uniform(1., 120., n)
    purpose = rng.integers(0, 3, n)
    state = rng.integers(0, 5, n)
    labels = np.array(['C', 'P', 'N'])[purpose]
    y = .008 * (.8 + .3 * age / 120.) * np.array([.8, 1.2, 1.])[purpose] * (1. + .03 * state)
    orig = pd.DataFrame({'age': age, 'purpose': pd.Categorical(labels),
                         'state': pd.Categorical(state), 'y': y})
    matrices = {
        'purpose': np.eye(3, dtype=dtype)[purpose],
        'state': np.eye(6, dtype=dtype)[state],
        'age': np.column_stack((np.ones(n), np.minimum(age, 36.) / 36.,
                                np.maximum(age - 36., 0.) / 84.)).astype(dtype),
    }
    groups = {name: [f'{name}_{i}' for i in range(X.shape[1])] for name, X in matrices.items()}
    expanded = pd.DataFrame(np.column_stack(list(matrices.values())),
                            columns=sum(groups.values(), []))
    container = ProductModelDataContainer(orig, expanded, response=y, feature_groups=groups,
                                         as_float32=dtype == np.float32)
    return container, groups, rng.uniform(.1, 2., n)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('weighted', [False, True])
def test_full_fit_matches_dense_with_age_purpose_and_pickle(dtype, weighted):
    data, groups, weights = make_fit_data(dtype)
    weights = weights if weighted else None
    options = dict(feature_groups=groups, interactions={'age': 'purpose'}, loss='poisson',
                   weights=weights, n_iterations=8, early_stopping_rounds=None, ftol=None, verbose=0)
    dense = DenseReference().fit(data, **options)
    fast = LinearProductRegressorBCD().fit(data, **options)
    tolerance = dict(rtol=1e-5, atol=1e-7) if dtype == np.float32 else dict(rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fast.predict(data), dense.predict(data), **tolerance)
    np.testing.assert_allclose(fast.loss_history_, dense.loss_history_, **tolerance)
    np.testing.assert_allclose(fast.global_scalar_, dense.global_scalar_, **tolerance)
    for group in groups:
        np.testing.assert_allclose(fast.block_means_[group], dense.block_means_[group], **tolerance)
        if group == 'age':
            for category in fast.coef_[group].categories:
                np.testing.assert_allclose(fast.coef_[group].categories[category],
                                           dense.coef_[group].categories[category], **tolerance)
        else:
            np.testing.assert_allclose(fast.normalized_coef_[group], dense.normalized_coef_[group], **tolerance)
    restored = pickle.loads(pickle.dumps(fast))
    np.testing.assert_array_equal(restored.predict(data), fast.predict(data))
    assert restored.fit_timing_ == fast.fit_timing_
    assert not any(isinstance(value, np.ndarray) and value.shape == (len(data.orig),)
                   for value in vars(restored).values())


def test_timing_resets_and_mse_frozen_and_interaction_blocks_stay_dense():
    data, groups, weights = make_fit_data()
    model = LinearProductRegressorBCD()
    blocks = data.get_expanded_array_dict(list(groups))
    model.loss_ = 'poisson'
    model.submodels_ = {'state': object()}
    codes = model._init_one_hot_codes(blocks, {'state': np.ones(6), 'purpose': np.ones(3)})
    assert set(codes) == {'purpose'}
    model.loss_ = 'mse'
    assert model._init_one_hot_codes(blocks, {'purpose': np.ones(3)}) == {}

    options = dict(feature_groups=groups, interactions={'age': 'purpose'}, weights=weights,
                   loss='poisson', early_stopping_rounds=None, ftol=None, verbose=0)
    model.fit(data, n_iterations=4, **options)
    previous = model.fit_timing_
    model.fit(data, n_iterations=2, **options)
    timing = model.fit_timing_
    assert timing is not previous
    assert len(timing['sweep_seconds']) == len(model.loss_history_) == 2
    assert set(timing['block_seconds']) == set(groups)
    assert timing['setup_seconds'] >= 0.
    assert all(value >= 0. for value in timing['block_seconds'].values())
    assert sum(timing['block_seconds'].values()) <= sum(timing['sweep_seconds'])
    assert timing['setup_seconds'] + sum(timing['sweep_seconds']) <= timing['total_seconds']


def test_regular_dropped_and_multihot_blocks_fall_back():
    model = LinearProductRegressorBCD()
    model.loss_ = 'poisson'
    model.submodels_ = {}
    blocks = {'dropped': np.array([[0., 0.], [1., 0.], [0., 1.]]),
              'multihot': np.array([[1., 1.], [1., 0.], [0., 1.]]),
              'numeric': np.array([[1., 1.], [1., 2.], [1., 3.]])}
    assert model._init_one_hot_codes(blocks, {name: np.ones(2) for name in blocks}) == {}
