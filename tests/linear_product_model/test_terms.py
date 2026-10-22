"""Terms and ModelSpec: a spec compiles to the blocks the solvers fit, so its fits match the container path."""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from quantbullet.linear_product_model import (InteractionCoef, LinearProductModelToolkit, LinearProductRegressor,
                                              ModelSpec)
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
from quantbullet.linear_product_model.terms import (BasisTerm, ByTerm, CategoricalTerm, FrozenTerm, ModelData,
                                                    TentTerm)
from quantbullet.model.feature import DataType, Feature, FeatureRole, FeatureSpec
from quantbullet.preprocessing import FlatRampTransformer, TentTransformer

AGE_KNOTS = (12., 36., 60.)
TENT_KNOTS = (1., 12., 24., 60., 120.)
FORMS = [("identity", "mse"), ("identity", "poisson"), ("log", "poisson")]


class FrozenCurve:
    def predict(self, X):
        return 1. + .1 * np.asarray(X, dtype=float).ravel() / 120.


def frame(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"purpose": pd.Categorical(rng.choice(["C", "N", "P"], n)),
                       "state": pd.Categorical(rng.integers(0, 4, n)),
                       "age": rng.uniform(1., 120., n), "sato": rng.uniform(-1., 1., n)})
    rate = (.05 * np.exp(.6 * np.minimum(df["age"], 36.) / 36.)
            * df["purpose"].map({"C": .8, "N": 1.2, "P": 1.}).astype(float) * (1. + .1 * df["state"].astype(int)))
    df["y"] = rng.poisson(rate).astype(float)
    df["weight"] = rng.uniform(.5, 2., n)
    return df


def toolkit_data(df, age_transformer, raw=()):
    """The container path, built as ``fit_turnover`` builds it: toolkit transformers, float32 blocks."""
    config = {name: OneHotEncoder(drop=None, handle_unknown="error", dtype=np.float32) for name in ("purpose", "state")}
    config["age"] = age_transformer
    features = [Feature(name, DataType.CATEGORY, FeatureRole.MODEL_INPUT) for name in ("purpose", "state")]
    features += [Feature(name, DataType.FLOAT, FeatureRole.MODEL_INPUT) for name in ("age", *raw)]
    toolkit = LinearProductModelToolkit(FeatureSpec(features + [Feature("y", DataType.FLOAT, FeatureRole.TARGET)]),
                                        preprocess_config=config).fit(df)
    blocks = []
    for name, columns in toolkit.feature_groups.items():
        values = config[name].transform(df[[name]]) if name in config else df[[name]].to_numpy()
        values = values.toarray() if hasattr(values, "toarray") else values
        blocks.append(pd.DataFrame(np.asarray(values, dtype=np.float32), columns=columns))
    container = ProductModelDataContainer(df, pd.concat(blocks, axis=1), response=df["y"].to_numpy(),
                                          feature_groups=toolkit.feature_groups, as_float32=True)
    return toolkit, container


def fit(data, block_link, loss, **options):
    return LinearProductRegressor().fit(data, loss=loss, block_link=block_link, verbose=0, **options)


def assert_same_fit(model, reference):
    assert list(model.coef_) == list(reference.coef_)
    for group, coef in reference.coef_.items():
        if isinstance(coef, InteractionCoef):
            assert model.coef_[group].by == coef.by
            for label, curve in coef.categories.items():
                np.testing.assert_array_equal(model.coef_[group].categories[label], curve)
        else:
            np.testing.assert_array_equal(model.coef_[group], coef)
    assert model.global_scalar_ == reference.global_scalar_
    assert model.loss_history_ == reference.loss_history_


@pytest.mark.parametrize("block_link, loss", FORMS)
@pytest.mark.parametrize("interaction", [False, True])
def test_a_spec_fits_exactly_as_the_container_path(block_link, loss, interaction):
    df = frame()
    toolkit, container = toolkit_data(df, FlatRampTransformer(knots=list(AGE_KNOTS), include_bias=True))
    reference = LinearProductRegressor().fit(container, toolkit.feature_groups, loss=loss, block_link=block_link,
                                             weights=df["weight"].to_numpy(), verbose=0,
                                             interactions={"age": "purpose"} if interaction else None)
    purpose, state = CategoricalTerm("purpose"), CategoricalTerm("state")
    age = BasisTerm("age", FlatRampTransformer(knots=list(AGE_KNOTS), include_bias=True))
    spec = ModelSpec([purpose, state, ByTerm(age, by=purpose, drop=("age",)) if interaction else age])
    data = spec.prepare(df, response="y", weights="weight")
    assert isinstance(data, ModelData) and data.feature_groups == toolkit.feature_groups
    model = fit(data, block_link, loss)
    assert_same_fit(model, reference)
    np.testing.assert_array_equal(model.predict(df), model.predict(data))
    np.testing.assert_array_equal(model.predict(data), reference.predict(container))


def test_prepare_leaves_the_spec_unfitted_and_the_model_keeps_a_fitted_copy():
    df = frame()
    tent, flat, purpose = TentTerm("age", TENT_KNOTS), BasisTerm("sato", FlatRampTransformer(knots=[0.])), CategoricalTerm("purpose")
    spec = ModelSpec([purpose, tent, flat])
    model = fit(spec.prepare(df, response="y"), "log", "poisson")
    assert not hasattr(tent, "transformer_") and not hasattr(purpose, "levels_")
    assert not hasattr(flat.transformer, "n_pieces_")
    assert model.spec_ is not spec and model.spec_.terms[0].levels_ == ["C", "N", "P"]
    assert model.spec_.feature_groups()["age"] == ["age_at_1", "age_at_12", "age_at_24", "age_at_60", "age_at_120"]
    new = frame(n=50, seed=9)
    np.testing.assert_array_equal(model.predict(new), model.predict(model.spec_.transform(new)))
    with pytest.raises(ValueError, match="needs a spec fitted by prepare"):
        spec.transform(new)


def test_term_references_match_reference_rows_of_the_toolkit():
    df = frame()
    toolkit, container = toolkit_data(df, TentTransformer(TENT_KNOTS))
    rows = toolkit.reference_rows({"age": 24, "purpose": "P", "state": 2})
    for interaction in (False, True):
        reference = LinearProductRegressor().fit(container, toolkit.feature_groups, loss="poisson", block_link="log",
                                                 interactions={"age": "purpose"} if interaction else None,
                                                 references=rows, verbose=0)
        purpose = CategoricalTerm("purpose", reference="P")
        age = TentTerm("age", TENT_KNOTS, reference=24)
        spec = ModelSpec([purpose, CategoricalTerm("state", reference=2),
                          ByTerm(age, by=purpose, drop=("age",)) if interaction else age])
        model = fit(spec.prepare(df, response="y"), "log", "poisson")
        assert_same_fit(model, reference)
        assert model.fit_metadata_.references == ["purpose", "state", "age"]


def test_a_frozen_term_is_a_frozen_submodel():
    df = frame()
    toolkit, container = toolkit_data(df, FlatRampTransformer(knots=list(AGE_KNOTS), include_bias=True), raw=("sato",))
    frozen = FrozenCurve()
    reference = LinearProductRegressor().fit(container, toolkit.feature_groups, submodels={"sato": frozen},
                                             loss="poisson", verbose=0)
    spec = ModelSpec([CategoricalTerm("purpose"), CategoricalTerm("state"),
                      BasisTerm("age", FlatRampTransformer(knots=list(AGE_KNOTS), include_bias=True)),
                      FrozenTerm("sato", frozen)])
    model = fit(spec.prepare(df, response="y"), "identity", "poisson")
    assert_same_fit(model, reference)


def test_categorical_terms_match_one_hot_encoding_and_reject_unknown_levels():
    df = frame()
    term = copy.deepcopy(CategoricalTerm("state"))
    basis = term.setup_basis(df, np.float32)
    encoder = OneHotEncoder(drop=None, dtype=np.float32).fit(df[["state"]])
    np.testing.assert_array_equal(basis, encoder.transform(df[["state"]]).toarray())
    assert term.columns() == encoder.get_feature_names_out(["state"]).tolist()
    with pytest.raises(ValueError, match=r"levels outside \[0, 1, 2, 3\]: \[7\]"):
        term.basis(pd.DataFrame({"state": [1, 7]}), np.float32)
    with pytest.raises(ValueError, match="purpose has missing values"):
        ModelSpec([CategoricalTerm("purpose")]).prepare(df.assign(purpose=df["purpose"].cat.add_categories("X")
                                                                  .where(df.index > 0)), response="y")
    named = CategoricalTerm("purpose", levels=["P", "C", "N"]).setup_basis(df, np.float64)
    np.testing.assert_array_equal(named[:, 0], (df["purpose"] == "P").to_numpy())


def test_spec_checks_run_before_any_data():
    purpose = CategoricalTerm("purpose")
    age = TentTerm("age", TENT_KNOTS)
    cases = [
        ([purpose, purpose], ValueError, r"share the names \['purpose'\]"),
        ([purpose, "age"], TypeError, "ModelSpec takes terms"),
        ([purpose, TentTerm("age", TENT_KNOTS, reference=[24, 60])], NotImplementedError, "several references"),
        ([purpose, ByTerm(age, by=CategoricalTerm("purpose"), drop=("age",))], ValueError, "same object"),
        ([purpose, ByTerm(age, by=purpose)], ValueError, r"needs drop written out; for this model it is drop=\('age',\)"),
        ([purpose, ByTerm(age, by=purpose, drop="any")], ValueError, "dropping purpose's reference .* constraint"),
        ([purpose, ByTerm(age, by=purpose, drop="all")], ValueError, "keeping age's reference .* collinear"),
        ([purpose, ByTerm(age, by=purpose, drop=("sato",))], ValueError, r"drop names \['sato'\]"),
    ]
    for terms, error, message in cases:
        with pytest.raises(error, match=message):
            ModelSpec(terms).check()
    assert ModelSpec([purpose, ByTerm(age, by=purpose, drop=("age",))]).check() is not None


def test_fit_takes_groups_from_the_spec_or_the_arguments_but_not_both():
    df = frame()
    data = ModelSpec([CategoricalTerm("purpose")]).prepare(df, response="y", weights="weight")
    for arguments, message in [(dict(feature_groups={"purpose": ["purpose_C"]}), "feature_groups come from the spec"),
                               (dict(weights=np.ones(len(df))), "weights come from the spec")]:
        with pytest.raises(ValueError, match=message):
            fit(data, "log", "poisson", **arguments)
    _, container = toolkit_data(df, FlatRampTransformer(knots=list(AGE_KNOTS), include_bias=True))
    with pytest.raises(ValueError, match="fit needs feature_groups"):
        fit(container, "log", "poisson")
