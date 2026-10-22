"""TentTransformer: the degree-1 B-spline basis whose coefficients are the curve's values at the knots."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.preprocessing import SplineTransformer

from quantbullet.preprocessing import TentTransformer


def test_basis_matches_the_hand_computed_table():
    x = pd.Series([-.5, .4, 1., 2.75, 5.], name="x")
    basis = TentTransformer(knots=[0, 1, 2, 3]).fit(x).transform(x)
    expected = [[1, 0, 0, 0], [.6, .4, 0, 0], [0, 1, 0, 0], [0, 0, .25, .75], [0, 0, 0, 1]]
    np.testing.assert_allclose(basis, expected, atol=1e-15)


def test_curve_passes_through_knot_values_is_linear_between_and_flat_outside():
    knots = np.array([-4., -2., -1.5, -.75])
    coef = np.array([.5, 1.2, 1.4, 2.])
    rng = np.random.default_rng(0)
    x = rng.uniform(-6., 1., 1000)
    transformer = TentTransformer(knots).fit(x[:, None])
    basis = transformer.transform(x[:, None])
    assert np.all(basis >= 0) and np.allclose(basis.sum(axis=1), 1.)
    np.testing.assert_allclose(transformer.transform(knots[:, None]) @ coef, coef)
    np.testing.assert_allclose(basis @ coef, np.interp(x, knots, coef))
    np.testing.assert_allclose(transformer.transform(np.array([[-100.], [100.]])) @ coef, [coef[0], coef[-1]])


def test_equals_sklearn_degree_one_spline_with_constant_extrapolation():
    knots = np.array([1., 3., 6., 12., 36., 120.])
    x = np.random.default_rng(1).uniform(0., 130., (500, 1))
    sklearn_basis = SplineTransformer(degree=1, knots=knots[:, None], extrapolation="constant").fit(x).transform(x)
    np.testing.assert_allclose(TentTransformer(knots).fit(x).transform(x), sklearn_basis, atol=1e-12)


def test_column_names_carry_the_knots_for_every_input_type():
    knots = (-.75, 24, 125000)
    values = [-1., 24., 200000.]
    expected = ["c_at_-0.75", "c_at_24", "c_at_125000"]
    for X, names in [(pd.Series(values, name="c"), expected), (pd.DataFrame({"c": values}), expected),
                     (np.array(values)[:, None], [name.replace("c_", "x0_") for name in expected])]:
        transformer = TentTransformer(knots).fit(X)
        assert transformer.get_feature_names_out().tolist() == names
        assert transformer.transform(X).shape == (3, 3)


def test_knots_are_checked_by_fit_and_the_transformer_clones():
    x = pd.Series([0., 1., 2., 3.], name="x")
    unfitted = TentTransformer(knots=[2, 1])                        # constructing does not check
    for knots, message in [([2, 1], "strictly increasing"), ([1], "at least two"), ([1, np.nan], "finite")]:
        with pytest.raises(ValueError, match=message):
            TentTransformer(knots).fit(x)
    assert clone(unfitted).get_params() == {"knots": [2, 1]}


def test_knots_without_nearby_data_and_non_finite_inputs_are_rejected():
    with pytest.raises(ValueError, match=r"x: no values between the neighbours of knots \[2.0\]"):
        TentTransformer(knots=[0, 1, 2, 3, 4]).fit(pd.Series([0., .5, 3.5, 4.], name="x"))
    with pytest.raises(ValueError, match="NaN"):
        TentTransformer(knots=[0, 1]).fit(pd.Series([0., np.nan, 1.]))
    transformer = TentTransformer(knots=[0, 1]).fit(pd.Series([0., 1.]))
    with pytest.raises(ValueError, match="exactly one feature"):
        transformer.transform(pd.DataFrame({"a": [0.], "b": [1.]}))
