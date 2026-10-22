import unittest
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error

from quantbullet.linear_product_model import (
    LinearProductRegressor,
    LinearProductRegressorScipy,
    ModelSpec,
)
from quantbullet.linear_product_model.terms import BasisTerm, CategoricalTerm, FrozenTerm, TentTerm
from quantbullet.preprocessing import FlatRampTransformer
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
from quantbullet.parametric_model import InterpolatedModel

def _setup_frame():
    np.random.seed(42)
    n_samples = int( 10e4 )
    df = pd.DataFrame({
        'x1': 10 + 3 * np.random.randn(n_samples),
        'x2': 20 + 3 * np.random.randn(n_samples),
        'x3': np.random.choice([0, 1, 2], size=n_samples, p=[0.25, 0.25, 0.5])
    })

    df['x3'] = df['x3'].astype('category')

    df['y'] = np.cos(df['x1']) + np.sin(df['x2']) + \
        np.random.randn(n_samples) * 0.5 + \
        np.random.normal(loc = df['x3'].cat.codes, scale=0.5) + 10

    return df


def _setup_spec( x1=None ):
    """FlatRamp x1 and x2 and one-hot x3; ``x1`` replaces the x1 term, e.g. with a ``FrozenTerm``."""
    return ModelSpec( [
        x1 if x1 is not None else BasisTerm( 'x1', FlatRampTransformer( knots=list( np.arange( 4, 16, 1 ) ), include_bias=True ) ),
        BasisTerm( 'x2', FlatRampTransformer( knots=list( np.arange( 14, 26, 1 ) ), include_bias=True ) ),
        CategoricalTerm( 'x3' ),
    ] )

def _setup_data():
    df = _setup_frame()
    data = _setup_spec().prepare( df, response='y', dtype=np.float64 )
    return df, data.expanded_df, data.feature_groups

class TestLinearProductRegressor(unittest.TestCase):
    def setUp(self):
        df, expanded_df, feature_groups = _setup_data()
        self.df = df
        self.train_df = expanded_df
        self.feature_groups = feature_groups

    def test_LinearProductRegressor(self):
        df = self.df
        train_df = self.train_df
        feature_groups = self.feature_groups

        dcontainer = ProductModelDataContainer( df, train_df, response= df['y'], feature_groups=feature_groups )
        lprm_ols = LinearProductRegressor()
        lprm_ols.fit( dcontainer, feature_groups=feature_groups, n_iterations=100, early_stopping_rounds=20, cache_qr_decomp=False )
        df['model_pred_BCD'] = lprm_ols.predict( dcontainer )

        # check the mean squared error; whether the model converges
        mse = mean_squared_error(df['y'], df['model_pred_BCD'])
        self.assertTrue( np.isclose(mse, 0.52, atol=0.01), f"MSE is {mse}, expected around 0.52" )
        
        # check whether the global scale converges to the mean of y
        self.assertTrue( abs( lprm_ols.global_scalar_ / df['y'].mean() -1 ) < 0.05 )

    def test_LinearProductRegressor_weights( self ):
        df = self.df
        train_df = self.train_df
        feature_groups = self.feature_groups

        dcontainer = ProductModelDataContainer( df, train_df, response= df['y'], feature_groups=feature_groups )
        lprm_ols = LinearProductRegressor()
        lprm_ols.fit( dcontainer, feature_groups=feature_groups, n_iterations=100, early_stopping_rounds=20, cache_qr_decomp=False, weights=np.array([1.0] * df.shape[0]) )
        df['model_pred_BCD'] = lprm_ols.predict( dcontainer )

        # check the mean squared error; whether the model converges
        mse = mean_squared_error(df['y'], df['model_pred_BCD'])
        self.assertTrue( np.isclose(mse, 0.52, atol=0.01), f"MSE is {mse}, expected around 0.52" )
        
        # check whether the global scale converges to the mean of y
        self.assertTrue( abs( lprm_ols.global_scalar_ / df['y'].mean() -1 ) < 0.05 )

class TestFrozenSubmodelBlocks(unittest.TestCase):
    """A block passed through ``submodels`` is held fixed during BCD.

    Every read path has to route through the submodel rather than the placeholder
    coefficient that init leaves behind in ``coef_``, otherwise fitting and
    prediction silently disagree about what that block contributes.
    """

    def setUp(self):
        self.df = _setup_frame()

    def _fit_with_frozen_x1(self):
        df = self.df
        grid = np.linspace( df['x1'].min(), df['x1'].max(), 40 )
        submodel = InterpolatedModel().fit( grid, 1.0 + 0.05 * np.cos( grid ) )

        # A frozen term's block is the raw column, which is the input the submodel expects.
        dcontainer = _setup_spec( FrozenTerm( 'x1', submodel ) ).prepare( df, response='y', dtype=np.float64 )
        self.assertEqual( dcontainer.feature_groups['x1'], ['x1'] )
        model = LinearProductRegressor()
        model.fit( dcontainer, n_iterations=20, early_stopping_rounds=5 )
        return model, dcontainer, submodel

    def test_block_mean_reads_from_submodel(self):
        model, dcontainer, submodel = self._fit_with_frozen_x1()
        block = dcontainer.get_expanded_array_for_feature_group('x1')

        self.assertAlmostEqual(
            model.block_means_['x1'], float( np.mean( submodel.predict( block ) ) ), places=10 )

    def test_frozen_block_prediction_matches_submodel(self):
        model, dcontainer, submodel = self._fit_with_frozen_x1()
        block = dcontainer.get_expanded_array_for_feature_group('x1')

        self.assertTrue( np.allclose(
            model.single_feature_group_predict('x1', dcontainer), submodel.predict( block ) ) )

    def test_full_prediction_factors_through_submodel(self):
        model, dcontainer, submodel = self._fit_with_frozen_x1()
        block = dcontainer.get_expanded_array_for_feature_group('x1')

        # Dividing the full prediction by every other block must leave exactly the
        # submodel curve times the global scalar.
        others = model.leave_out_feature_group_predict('x1', dcontainer)
        implied = model.predict( dcontainer ) / others

        self.assertTrue( np.allclose( implied, submodel.predict( block ), rtol=1e-6 ) )


class TestLinearProductRegressorScipy(unittest.TestCase):
    def setUp(self):
        df, expanded_df, feature_groups = _setup_data()
        self.df = df
        self.train_df = expanded_df
        self.feature_groups = feature_groups

    def test_LinearProductRegressorScipy(self):
        df = self.df
        train_df = self.train_df
        feature_groups = self.feature_groups

        lpm_scipy = LinearProductRegressorScipy( xtol=1e-12, gtol=1e-12, ftol=1e-12 )
        lpm_scipy.fit( X=train_df, y=df['y'], feature_groups=feature_groups, verbose=1 )
        df['model_predict_scipy'] = lpm_scipy.predict(train_df)

        # check the mean squared error
        mse = mean_squared_error(df['y'], df['model_predict_scipy'])
        self.assertTrue( np.isclose(mse, 0.52, atol=0.01), f"MSE is {mse}, expected around 0.52" )

    def test_leave_out_feature_group_predict(self):
        df = self.df
        train_df = self.train_df
        feature_groups = self.feature_groups

        lprm_ols = LinearProductRegressorScipy()
        lprm_ols.fit( train_df, df['y'], feature_groups=feature_groups )

        # leave out x1
        df['model_pred_x1_excluded'] = lprm_ols.leave_out_feature_group_predict('x1', train_df)

        # leave out x2
        df['model_pred_x2_excluded'] = lprm_ols.leave_out_feature_group_predict('x2', train_df)

        # leave out x3
        df['model_pred_x3_excluded'] = lprm_ols.leave_out_feature_group_predict('x3', train_df)

        # check the predictions and product of predictions
        df['model_pred_product'] = np.sqrt( df['model_pred_x1_excluded'] * df['model_pred_x2_excluded'] * df['model_pred_x3_excluded'] )
        df['model_pred'] = lprm_ols.predict(train_df)

        # check two predictions should be close element wise
        self.assertTrue(np.allclose(df['model_pred'], df['model_pred_product'], atol=0.01),
                        "Predictions with and without feature groups should be close.")


# ----------------------------------------------------------------------
# references: blocks rescaled to 1 at a reference row
# ----------------------------------------------------------------------

AGE_KNOTS = (1., 12., 24., 60., 120.)
FORMS = [("identity", "mse"), ("identity", "poisson"), ("log", "poisson")]


class FrozenCurve:
    def __init__(self, coef):
        self.coef = np.asarray(coef, dtype=float)

    def predict(self, X):
        return X @ self.coef


def _reference_data(n=6000, seed=3):
    """Counts driven by age (tent basis), purpose and state (one-hot), as a container built from a spec."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"age": rng.uniform(1., 120., n),
                       "purpose": pd.Categorical(rng.choice(["C", "N", "P"], n)),
                       "state": pd.Categorical(rng.integers(0, 4, n))})
    rate = (.05 * np.exp(.6 * np.minimum(df["age"], 36.) / 36.)
            * df["purpose"].map({"C": .8, "N": 1.2, "P": 1.}).astype(float) * (1. + .1 * df["state"].astype(int)))
    df["y"] = rng.poisson(rate).astype(float)
    spec = ModelSpec([TentTerm("age", AGE_KNOTS), CategoricalTerm("purpose"), CategoricalTerm("state")])
    prepared = spec.prepare(df, response="y", dtype=np.float64)
    return df, prepared.spec, _container(df, prepared, prepared.feature_groups)


def _container(df, prepared, feature_groups):
    """The container path: the spec's design without its spec."""
    return ProductModelDataContainer(df, prepared.expanded_df, response=df["y"].to_numpy(),
                                     feature_groups=feature_groups)


def _fit(container, groups, block_link, loss, **options):
    return LinearProductRegressor().fit(container, groups, loss=loss, block_link=block_link, verbose=0, **options)


@pytest.mark.parametrize("block_link, loss", FORMS)
@pytest.mark.parametrize("interaction", [False, True])
def test_references_move_levels_without_changing_predictions(block_link, loss, interaction):
    df, spec, data = _reference_data()
    groups, interactions = spec.feature_groups(), {"age": "purpose"} if interaction else None
    rows = spec.reference_rows({"age": 24, "purpose": "P", "state": 2})
    # Newton fixes the reference columns instead of its own pivots: a float64 Hessian takes both fits to the optimum.
    options = {"solver_options": {"design_cache": False}} if block_link == "log" else {}
    plain = _fit(data, groups, block_link, loss, interactions=interactions, **options)
    model = _fit(data, groups, block_link, loss, interactions=interactions, references=rows, **options)
    np.testing.assert_allclose(model.predict(data), plain.predict(data), rtol=1e-10)

    one = 1. if block_link == "identity" else 0.               # the coefficient whose factor is 1
    knot, purpose_p, state_2 = AGE_KNOTS.index(24.), groups["purpose"].index("purpose_P"), groups["state"].index("state_2")
    curves = model.coef_["age"].categories.values() if interaction else [model.coef_["age"]]
    for coef in curves:
        assert coef[knot] == pytest.approx(one, abs=1e-12)
    assert model.coef_["purpose"][purpose_p] == pytest.approx(one, abs=1e-12)
    assert model.coef_["state"][state_2] == pytest.approx(one, abs=1e-12)

    # Every factor is 1 at the reference state, so the scalar is the prediction there.
    state = pd.DataFrame({"age": [24.], "purpose": pd.Categorical(["P"]), "state": pd.Categorical([2])})
    single = spec.transform(state)
    assert model.predict(single)[0] == pytest.approx(model.global_scalar_, rel=1e-12)
    assert model.global_scalar_ == model.global_scalar_history_[model.best_iteration_]
    for params in model.coef_history_:
        assert params["purpose"][purpose_p] == pytest.approx(one, abs=1e-12)
    assert model.block_means_["purpose"] == pytest.approx(np.mean(model.single_feature_group_predict("purpose", data)))
    assert list(model.references_) == model.fit_metadata_.references == ["age", "purpose", "state"]


def test_a_reference_off_the_knots_leaves_other_blocks_at_mean_one():
    df, spec, data = _reference_data()
    rows = spec.reference_rows({"age": 30.})
    model = _fit(data, spec.feature_groups(), "log", "poisson", references=rows)
    assert model.block_factor("age", rows["age"][None, :])[0] == pytest.approx(1., rel=1e-12)
    for group in ("purpose", "state"):
        assert model.block_means_[group] == pytest.approx(1., abs=1e-10)


def test_bad_references_are_rejected_before_the_model_changes():
    df, spec, data = _reference_data()
    groups = spec.feature_groups()
    model = _fit(data, groups, "identity", "poisson")
    before = model.predict(data)
    age = spec.reference_rows({"age": 24})["age"]
    without_first = {**groups, "purpose": groups["purpose"][1:], "age": groups["age"][1:]}
    cases = [
        (data, groups, dict(references=[age]), TypeError, "references must be a dict"),
        (data, groups, dict(references={"income": age}), ValueError, "'income' is not in feature_groups"),
        (data, groups, dict(references={"age": age[:2]}), ValueError, "row of 'age' must be 5 finite values"),
        (data, groups, dict(references={"age": np.full(5, np.nan)}), ValueError, "must be 5 finite values"),
        (data, groups, dict(references={"state": np.eye(4)[2]}, submodels={"state": FrozenCurve(np.ones(4))}),
         ValueError, "'state' is a frozen submodel"),
        (data, groups, dict(references={"age": age}, interactions={"age": "purpose"},
                            interaction_submodels={"age": {"C": FrozenCurve(np.ones(5))}}),
         ValueError, "frozen interaction curves"),
        (_container(df, data, without_first), without_first,
         dict(references={"age": np.eye(4)[1]}, interactions={"age": "purpose"}),
         ValueError, "by-block 'purpose' to be a trainable full one-hot block"),
        (_container(df, data, without_first), without_first,
         dict(references={"age": np.eye(4)[1]}, block_link="log"), ValueError, "'age' has no constant direction"),
    ]
    for container, fit_groups, arguments, error, message in cases:
        with pytest.raises(error, match=message):
            model.fit(container, fit_groups, **{"loss": "poisson", "verbose": 0, **arguments})
        np.testing.assert_array_equal(model.predict(data), before)


def test_a_non_positive_reference_factor_keeps_the_fit_without_references():
    df, spec, data = _reference_data()
    plain = _fit(data, spec.feature_groups(), "identity", "poisson")
    model = LinearProductRegressor()
    with pytest.raises(ValueError, match="reference factor of 'age' is 0 in the fitted model"):
        model.fit(data, spec.feature_groups(), loss="poisson", verbose=0, references={"age": np.zeros(5)})
    np.testing.assert_array_equal(model.predict(data), plain.predict(data))
    assert model.references_ == {} and model.fit_metadata_.references is None