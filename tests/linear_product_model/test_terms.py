"""Terms and ModelSpec: a spec compiles to the blocks the solvers fit, so its fits match the container path."""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from quantbullet.linear_product_model import (InteractionCoef, LinearProductRegressor,
                                              ModelSpec)
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
from quantbullet.linear_product_model.terms import (BasisTerm, ByTerm, CategoricalTerm, FrozenTerm, ModelData,
                                                    TensorTerm, TentTerm)
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


def container_data(df, age_transformer, raw=()):
    """The container path as ``fit_turnover`` built it before specs: a transformer per column, float32 blocks.

    Returns the fitted transformers, the feature groups and the container; ``raw`` columns are their own block.
    """
    transformers = {name: OneHotEncoder(drop=None, handle_unknown="error", dtype=np.float32)
                    for name in ("purpose", "state")}
    transformers["age"] = age_transformer
    groups, blocks = {}, []
    for name in ("purpose", "state", "age", *raw):
        if name in transformers:
            # A tent basis is built in float32 directly, as the spec builds it, so its rows sum to exactly 1.
            exact = {"dtype": np.float32} if isinstance(transformers[name], TentTransformer) else {}
            values = transformers[name].fit_transform(df[[name]], **exact)
            groups[name] = transformers[name].get_feature_names_out([name]).tolist()
        else:
            values, groups[name] = df[[name]].to_numpy(), [name]
        values = values.toarray() if hasattr(values, "toarray") else values
        blocks.append(pd.DataFrame(np.asarray(values, dtype=np.float32), columns=groups[name]))
    container = ProductModelDataContainer(df, pd.concat(blocks, axis=1), response=df["y"].to_numpy(),
                                          feature_groups=groups, as_float32=True)
    return transformers, groups, container


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
    _, groups, container = container_data(df, FlatRampTransformer(knots=list(AGE_KNOTS), include_bias=True))
    reference = LinearProductRegressor().fit(container, groups, loss=loss, block_link=block_link,
                                             weights=df["weight"].to_numpy(), verbose=0,
                                             interactions={"age": "purpose"} if interaction else None)
    purpose, state = CategoricalTerm("purpose"), CategoricalTerm("state")
    age = BasisTerm("age", FlatRampTransformer(knots=list(AGE_KNOTS), include_bias=True))
    spec = ModelSpec([purpose, state, ByTerm(age, by=purpose, drop=("age",)) if interaction else age])
    data = spec.prepare(df, response="y", weights="weight")
    assert isinstance(data, ModelData) and data.feature_groups == groups
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
    assert not hasattr(flat.transformer, "feature_names_out_")
    assert model.spec_ is not spec and model.spec_.terms[0].levels_ == ["C", "N", "P"]
    assert model.spec_.feature_groups()["age"] == ["age_at_1", "age_at_12", "age_at_24", "age_at_60", "age_at_120"]
    new = frame(n=50, seed=9)
    np.testing.assert_array_equal(model.predict(new), model.predict(model.spec_.transform(new)))
    with pytest.raises(ValueError, match="needs a spec fitted by prepare"):
        spec.transform(new)


def test_term_references_match_reference_rows_on_the_container_path():
    df = frame()
    transformers, groups, container = container_data(df, TentTransformer(TENT_KNOTS))
    rows = {}
    for name, value in {"age": 24, "purpose": "P", "state": 2}.items():
        row = transformers[name].transform(pd.DataFrame({name: [value]}))
        rows[name] = np.asarray(row.toarray() if hasattr(row, "toarray") else row, dtype=float).ravel()
    for interaction in (False, True):
        reference = LinearProductRegressor().fit(container, groups, loss="poisson", block_link="log",
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
    _, groups, container = container_data(df, FlatRampTransformer(knots=list(AGE_KNOTS), include_bias=True),
                                          raw=("sato",))
    frozen = FrozenCurve()
    reference = LinearProductRegressor().fit(container, groups, submodels={"sato": frozen},
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
        ([purpose, TentTerm("age", TENT_KNOTS, reference=[24, 30])], ValueError,
         r"several references must be knots \[1.0, 12.0, 24.0, 60.0, 120.0\]; \[30\] are not"),
        ([purpose, TentTerm("age", TENT_KNOTS, reference=[24, 24])], ValueError, "repeats a reference"),
        ([purpose, BasisTerm("sato", FlatRampTransformer(knots=[0.]), reference=[-1., 1.])], ValueError,
         "only TentTerm and CategoricalTerm"),
        ([CategoricalTerm("purpose", reference=["P", "N"]),], None, None),
        ([purpose, ByTerm(age, by=CategoricalTerm("purpose"), drop=("age",))], ValueError, "same object"),
        ([purpose, ByTerm(age, by=purpose)], ValueError, r"needs drop written out; for this model it is drop=\('age',\)"),
        ([purpose, ByTerm(age, by=purpose, drop="any")], ValueError,
         "removes the part that varies only with age, which no other term provides, so the model gains a constraint"),
        ([purpose, ByTerm(age, by=purpose, drop="all")], ValueError,
         r"keeps the part that varies only with purpose, which CategoricalTerm\('purpose'\) also provides, so the model "
         "is collinear"),
        ([purpose, ByTerm(age, by=purpose, drop=("sato",))], ValueError, r"drop names \['sato'\]"),
    ]
    for terms, error, message in cases:
        if error is None:
            ModelSpec(terms).check()
            continue
        with pytest.raises(error, match=message):
            ModelSpec(terms).check()
    tied_by = CategoricalTerm("purpose", reference=["P", "N"])
    with pytest.raises(ValueError, match="by term of a ByTerm.*give it one reference"):
        ModelSpec([tied_by, ByTerm(age, by=tied_by, drop=("age",))]).check()
    assert ModelSpec([purpose, ByTerm(age, by=purpose, drop=("age",))]).check() is not None


def test_fit_takes_groups_from_the_spec_or_the_arguments_but_not_both():
    df = frame()
    data = ModelSpec([CategoricalTerm("purpose")]).prepare(df, response="y", weights="weight")
    for arguments, message in [(dict(feature_groups={"purpose": ["purpose_C"]}), "feature_groups come from the spec"),
                               (dict(weights=np.ones(len(df))), "weights come from the spec")]:
        with pytest.raises(ValueError, match=message):
            fit(data, "log", "poisson", **arguments)
    _, _, container = container_data(df, FlatRampTransformer(knots=list(AGE_KNOTS), include_bias=True))
    with pytest.raises(ValueError, match="fit needs feature_groups"):
        fit(container, "log", "poisson")


def test_reference_rows_put_raw_values_through_the_fitted_terms():
    df = frame()
    spec = ModelSpec([TentTerm("age", TENT_KNOTS), CategoricalTerm("purpose"),
                      BasisTerm("sato", FlatRampTransformer(knots=[0.], include_bias=True))])
    fitted = spec.prepare(df, response="y").spec
    rows = fitted.reference_rows({"age": 30., "purpose": "P", "sato": .5})
    np.testing.assert_allclose(rows["age"], [0, 0, 5 / 6, 1 / 6, 0])
    np.testing.assert_array_equal(rows["purpose"], [0, 0, 1])
    np.testing.assert_allclose(rows["sato"], fitted.term_dict()["sato"].transformer.transform(
        pd.DataFrame({"sato": [.5]}))[0])
    with pytest.raises(ValueError, match=r"No terms \['income'\]"):
        fitted.reference_rows({"income": 1.})
    with pytest.raises(ValueError, match="needs a spec fitted by prepare"):
        spec.reference_rows({"age": 30.})


@pytest.mark.parametrize("block_link", ["identity", "log"])
def test_model_terms_read_coefficients_curves_and_tables(block_link):
    df = frame()
    purpose = CategoricalTerm("purpose", reference="P")
    sato = BasisTerm("sato", FlatRampTransformer(knots=[0.], include_bias=True))
    spec = ModelSpec([purpose, ByTerm(TentTerm("age", TENT_KNOTS, reference=24), by=purpose, drop=("age",)), sato])
    model = fit(spec.prepare(df, response="y"), block_link, "poisson")
    one = 1. if block_link == "identity" else 0.

    age = model.term("age")
    assert list(age.coef.columns) == ["C", "N", "P"] and list(age.coef.index) == model.spec_.feature_groups()["age"]
    table = age.table()
    assert list(table.index) == list(TENT_KNOTS) and list(table.columns) == ["C", "N", "P"]
    np.testing.assert_allclose(table.loc[24.], 1., rtol=1e-12)
    np.testing.assert_allclose(age.curve([12., 60.], level="N"), table.loc[[12., 60.], "N"].to_numpy())
    with pytest.raises(ValueError, match="needs level= one of"):
        age.curve([12.])

    levels = model.term("purpose")
    assert levels.coef["purpose_P"] == pytest.approx(one, abs=1e-12)
    np.testing.assert_allclose(levels.table().to_numpy(), model.term("purpose").curve(["C", "N", "P"]))
    assert levels.table()["P"] == pytest.approx(1., rel=1e-12)

    x = np.array([-.5, 0., .5])
    np.testing.assert_array_equal(model.term("sato").curve(x), model.block_factor(
        "sato", model.spec_.term_dict()["sato"].basis(pd.DataFrame({"sato": x}), float)))
    with pytest.raises(ValueError, match="no knots or levels"):
        model.term("sato").table()
    with pytest.raises(KeyError, match="No term 'income'"):
        model.term("income")
    _, groups, container = container_data(df, FlatRampTransformer(knots=list(AGE_KNOTS), include_bias=True))
    with pytest.raises(ValueError, match="needs a model fitted on data from ModelSpec.prepare"):
        LinearProductRegressor().fit(container, groups, loss="poisson", verbose=0).term("age")


def test_describe_lines_name_the_basis_and_the_reference():
    purpose = CategoricalTerm("purpose", reference="P")
    age = TentTerm("age", (1, 12, 120), reference=24)
    assert age.describe() == "knots=[1, 12, 120], reference=24"
    assert purpose.describe() == "levels from data, reference='P'"
    assert BasisTerm("sato", FlatRampTransformer(knots=[0.])).describe() == "FlatRampTransformer knots=[0]"
    assert ByTerm(age, by=purpose, drop=("age",)).describe() == (
        "knots=[1, 12, 120], reference=24; one curve per purpose level, drop=('age',)")
    fitted = ModelSpec([purpose]).prepare(frame(), response="y").spec
    assert fitted.terms[0].describe() == "3 levels, reference='P'"


def tied_spec(age_reference=(24., 60.), purpose_reference=("P", "N")):
    return ModelSpec([CategoricalTerm("purpose", reference=list(purpose_reference)), CategoricalTerm("state"),
                      TentTerm("age", TENT_KNOTS, reference=list(age_reference))])


@pytest.mark.parametrize("block_link", ["identity", "log"])
def test_several_references_tie_their_columns_and_equal_the_hand_merged_design(block_link):
    df = frame()
    model = fit(tied_spec().prepare(df, response="y", weights="weight"), block_link, "poisson")
    groups = model.spec_.feature_groups()
    assert groups["age"] == ["age_at_1", "age_at_12", "age_at_24+60", "age_at_120"]
    assert groups["purpose"] == ["purpose_C", "purpose_P+N"]
    np.testing.assert_allclose(model.term("age").table().loc[[24., 60.]], 1., rtol=1e-12)
    np.testing.assert_allclose(model.term("purpose").table().loc[["P", "N"]], 1., rtol=1e-12)

    # The same design merged by hand on the container path, with unit reference rows.
    tent = TentTransformer(TENT_KNOTS).fit_transform(df[["age"]], dtype=np.float32)
    age = np.column_stack([tent[:, 0], tent[:, 1], tent[:, 2] + tent[:, 3], tent[:, 4]])
    purpose = np.column_stack([(df["purpose"] == "C").to_numpy(), df["purpose"].isin(["P", "N"]).to_numpy()])
    state = OneHotEncoder(drop=None).fit_transform(df[["state"]]).toarray()
    blocks = {"purpose": purpose, "state": state, "age": age}
    expanded = pd.concat([pd.DataFrame(np.asarray(X, dtype=np.float32), columns=groups[name])
                          for name, X in blocks.items()], axis=1)
    container = ProductModelDataContainer(df, expanded, response=df["y"].to_numpy(), feature_groups=groups)
    reference = LinearProductRegressor().fit(container, groups, loss="poisson", block_link=block_link,
                                             weights=df["weight"].to_numpy(), verbose=0,
                                             references={"purpose": np.eye(2)[1], "age": np.eye(4)[2]})
    assert_same_fit(model, reference)


def test_tied_references_under_log_equal_dropping_their_columns():
    df = frame()
    options = {"solver_options": {"design_cache": False}}
    model = fit(tied_spec().prepare(df, response="y"), "log", "poisson", **options)
    # Dropping a tent's knot columns or one-hot levels fixes them at 0 on the log scale, i.e. at a factor of 1.
    prepared = ModelSpec([CategoricalTerm("purpose"), CategoricalTerm("state"), TentTerm("age", TENT_KNOTS)]).prepare(
        df, response="y")
    groups = {name: [column for column in columns if column not in ("purpose_N", "purpose_P", "age_at_24", "age_at_60")]
              for name, columns in prepared.feature_groups.items()}
    dropped = LinearProductRegressor().fit(
        ProductModelDataContainer(df, prepared.expanded_df, response=df["y"].to_numpy(), feature_groups=groups),
        groups, loss="poisson", block_link="log", verbose=0, **options)
    np.testing.assert_allclose(model.predict(df), dropped.predict(
        ProductModelDataContainer(df, prepared.expanded_df, feature_groups=groups)), rtol=1e-10)


def test_newton_holds_the_reference_column_as_its_pivot():
    df = frame()
    options = {"solver_options": {"design_cache": False}}
    purpose = CategoricalTerm("purpose", reference="N")
    spec = ModelSpec([purpose, CategoricalTerm("state"), ByTerm(TentTerm("age", TENT_KNOTS, reference=24), by=purpose,
                                                                 drop=("age",))])
    model = fit(spec.prepare(df, response="y"), "log", "poisson", **options)
    pivots = model.convergence_info_["pivots"]
    assert pivots["purpose"] == {"column": "purpose_N", "from": "reference"}
    assert pivots["age"] == {"column": "age_at_24", "from": "reference"}
    assert pivots["state"]["from"] == "auto"

    plain = ModelSpec([CategoricalTerm("purpose"), CategoricalTerm("state"),
                       ByTerm(TentTerm("age", TENT_KNOTS), by=(by := CategoricalTerm("purpose")), drop=("age",))])
    plain.terms[0] = by
    unreferenced = fit(plain.prepare(df, response="y"), "log", "poisson", **options)
    assert unreferenced.convergence_info_["pivots"]["age"]["from"] == "auto"
    np.testing.assert_allclose(model.predict(df), unreferenced.predict(df), rtol=1e-10)

    off_knot = ModelSpec([CategoricalTerm("purpose"), TentTerm("age", TENT_KNOTS, reference=30)])
    assert fit(off_knot.prepare(df, response="y"), "log", "poisson").convergence_info_["pivots"]["age"]["from"] == "auto"
    tied = fit(tied_spec().prepare(df, response="y"), "log", "poisson").convergence_info_["pivots"]
    assert tied["age"] == {"column": "age_at_24+60", "from": "reference"}
    assert tied["purpose"] == {"column": "purpose_P+N", "from": "reference"}



# ----------------------------------------------------------------------
# tensor terms
# ----------------------------------------------------------------------

SATO_KNOTS = (-1., -.5, 0., .5, 1.)


def tensor_terms(age_reference=24., sato_reference=0.):
    return TentTerm("age", TENT_KNOTS, reference=age_reference), TentTerm("sato", SATO_KNOTS, reference=sato_reference)


@pytest.mark.parametrize("mains, correct", [(("age", "sato"), "any"), (("age",), ("sato",)), ((), "all")])
def test_a_tensor_term_takes_only_the_rule_its_model_determines(mains, correct):
    age, sato = tensor_terms()
    main_terms = [term for term in (age, sato) if term.name in mains]
    for drop in ("any", "all", ("age",), ("sato",), None):
        spec = ModelSpec([*main_terms, TensorTerm(age, sato, drop=drop)])
        if drop == correct:
            spec.check()
            continue
        message = "needs drop written out" if drop is None else "so the model (is collinear|gains a constraint)"
        with pytest.raises(ValueError, match=message) as raised:
            spec.check()
        assert f"For this model it is drop={correct!r}" in str(raised.value) or \
            f"for this model it is drop={correct!r}" in str(raised.value)


def test_tensor_terms_are_checked_before_any_data():
    age, sato = tensor_terms()
    cases = [
        (TensorTerm(age, BasisTerm("x", FlatRampTransformer(knots=[0.])), drop="all"), ValueError,
         "takes TentTerm and CategoricalTerm terms"),
        (TensorTerm(age, TentTerm("sato", SATO_KNOTS, reference=.2), drop="all"), ValueError, "not one of its knots"),
        (TensorTerm(age, TentTerm("sato", SATO_KNOTS), drop="all"), ValueError, "drops TentTerm\\('sato'\\)'s reference"),
        (TensorTerm(age, age, drop="all"), ValueError, "repeats a variable"),
        (TensorTerm(age, TentTerm("sato", SATO_KNOTS, reference=[0., .5]), drop="all"), ValueError,
         "has several references"),
    ]
    for term, error, message in cases:
        with pytest.raises(error, match=message):
            ModelSpec([term]).check()
    with pytest.raises(ValueError, match="at least two terms"):
        TensorTerm(age, drop="all")
    finer = TentTerm("sato", (-1., -.75, -.5, 0., .5, 1.), reference=0.)
    with pytest.warns(UserWarning, match="has knots that TentTerm\\('sato'\\) lacks"):
        ModelSpec([age, sato, TensorTerm(age, finer, drop="any")]).check()
    assert TensorTerm(age, sato, drop="all", name="age_by_sato").name == "age_by_sato"


def test_tensor_log_fit_equals_the_design_with_the_reference_cells_dropped():
    df = frame()
    options = {"solver_options": {"design_cache": False}}
    age, sato = tensor_terms()
    spec = ModelSpec([CategoricalTerm("purpose"), age, sato, TensorTerm(age, sato, drop="any")])
    data = spec.prepare(df, response="y", weights="weight", dtype=np.float64)
    assert "age×sato" in data.feature_groups and "age×sato" not in df.columns
    model = fit(data, "log", "poisson", **options)
    assert model.convergence_info_["pivots"]["age×sato"] == {"column": "age×sato:reference", "from": "reference"}

    # The same model with the reference row and column of the node table, and the main references, deleted.
    tent_a, tent_s = (TentTransformer(knots).fit_transform(df[[name]]) for name, knots in
                      (("age", TENT_KNOTS), ("sato", SATO_KNOTS)))
    ia, js = TENT_KNOTS.index(24.), SATO_KNOTS.index(0.)
    cells = {f"a{i}s{j}": tent_a[:, i] * tent_s[:, j] for i in range(len(TENT_KNOTS)) for j in range(len(SATO_KNOTS))
             if i != ia and j != js}
    blocks = {"purpose": pd.get_dummies(df["purpose"]).to_numpy(dtype=float),
              "age": np.delete(tent_a, ia, axis=1), "sato": np.delete(tent_s, js, axis=1),
              "age×sato": np.column_stack(list(cells.values()))}
    groups = {name: [f"{name}_{k}" for k in range(X.shape[1])] for name, X in blocks.items()}
    expanded = pd.concat([pd.DataFrame(X, columns=groups[name]) for name, X in blocks.items()], axis=1)
    deleted = LinearProductRegressor().fit(
        ProductModelDataContainer(df, expanded, response=df["y"].to_numpy(), feature_groups=groups, groups_in_orig=False),
        groups, loss="poisson", block_link="log", weights=df["weight"].to_numpy(), verbose=0, **options)
    np.testing.assert_allclose(model.predict(df), deleted.predict(
        ProductModelDataContainer(df, expanded, feature_groups=groups, groups_in_orig=False)), rtol=1e-10)


@pytest.mark.parametrize("block_link", ["identity", "log"])
def test_tensor_tables_are_one_on_the_reference_lines(block_link):
    df = frame()
    age, sato = tensor_terms()
    model = fit(ModelSpec([age, sato, TensorTerm(age, sato, drop="any")]).prepare(df, response="y", dtype=np.float64),
                block_link, "poisson")
    table = model.term("age×sato").table()
    assert list(table.index) == list(TENT_KNOTS) and list(table.columns) == list(SATO_KNOTS)
    np.testing.assert_allclose(table.loc[24.], 1., rtol=1e-12)
    np.testing.assert_allclose(table[0.], 1., rtol=1e-12)
    curve = model.term("age×sato").curve({"age": [12., 60.], "sato": [-.5, 1.]})
    np.testing.assert_allclose(curve, [table.loc[12., -.5], table.loc[60., 1.]], rtol=1e-12)


def test_a_categorical_tensor_term_matches_the_by_term():
    df = frame()
    options = {"solver_options": {"design_cache": False}}
    purpose = CategoricalTerm("purpose", reference="P")
    age = TentTerm("age", TENT_KNOTS, reference=24.)
    by = fit(ModelSpec([purpose, ByTerm(age, by=purpose, drop=("age",))]).prepare(df, response="y", dtype=np.float64),
             "log", "poisson", **options)
    tensor = fit(ModelSpec([purpose, TensorTerm(purpose, age, drop=("age",))]).prepare(df, response="y",
                                                                                      dtype=np.float64),
                 "log", "poisson", **options)
    np.testing.assert_allclose(tensor.predict(df), by.predict(df), rtol=1e-10)
    # Same references, so the same tables: each level's age curve is 1 at age 24, and P's level is 1.
    np.testing.assert_allclose(tensor.term("purpose×age").table().T.to_numpy(), by.term("age").table().to_numpy(),
                               rtol=1e-8)
    np.testing.assert_allclose(tensor.term("purpose").table(), by.term("purpose").table(), rtol=1e-8)


def test_the_toolkit_skips_tensor_terms_in_one_dimensional_implied_actuals():
    from quantbullet.linear_product_model import LinearProductModelToolkit
    df = frame()
    age, sato = tensor_terms()
    data = ModelSpec([age, sato, TensorTerm(age, sato, drop="any")]).prepare(df, response="y")
    model = fit(data, "log", "poisson")
    raw = LinearProductModelToolkit(model.spec_).compute_implied_actual_data(model, data, sample_frac=.5)
    assert sorted(raw) == ["age", "sato"]
    np.testing.assert_allclose(model.leave_out_feature_group_predict("age×sato", data) *
                               model.single_feature_group_predict("age×sato", data), model.predict(data), rtol=1e-12)
