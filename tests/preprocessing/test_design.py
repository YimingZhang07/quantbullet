"""Design: encodings and crosses with every choice written out, as one full-rank additive matrix."""
import numpy as np
import pandas as pd
import pytest

from quantbullet.preprocessing import Cross, Design, OneHot, Tent

AGE_KNOTS = (0, 12, 24, 60, 120)
INCENTIVE_KNOTS = (-3, -1, 0, 1, 3)


def frame(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"purpose": rng.choice(["C", "N", "P"], n), "age": rng.uniform(0, 120, n),
                         "incentive": rng.uniform(-3, 3, n), "ltv": rng.uniform(20, 97, n)})


def encodings():
    return (OneHot("purpose", reference="P"), Tent("age", AGE_KNOTS, reference=24),
            Tent("incentive", INCENTIVE_KNOTS, reference=0), Tent("ltv", (20, 80, 97), reference=80))


def rank(*matrices):
    return np.linalg.matrix_rank(np.hstack([np.asarray(m, dtype=float) for m in matrices]))


def every_column(design, df):
    """Each fitted term's columns before the design drops any, the constant included."""
    columns = [np.ones((len(df), 1))]
    for term in design.terms_:
        if isinstance(term, Cross):
            columns.append(Cross.product([encoding.basis(df, np.float64) for encoding in term.encodings]))
        else:
            columns.append(term.basis(df, np.float64))
    return np.hstack(columns)


def test_a_cell_is_dropped_where_any_condition_holds():
    df = frame()
    _, age, incentive, _ = encodings()
    both = Design([age, incentive, Cross(age, incentive, drop=["age", "incentive"])], intercept=True).fit(df)
    grid = both.terms_[-1].dropped().reshape(5, 5)
    assert grid[2].all() and grid[:, 2].all() and grid.sum() == 9            # the age=24 row and the incentive=0 column
    assert len(both.groups()["age_x_incentive"]) == 16
    alone = Design([Cross(age, incentive, drop=[("age", "incentive")])], intercept=True).fit(df)
    kept = alone.groups()["age_x_incentive"]
    assert len(kept) == 24 and "age_at_24_x_incentive_at_0" not in kept


def test_the_design_is_full_rank_and_drops_nothing_it_needs():
    df = frame()
    purpose, age, incentive, _ = encodings()
    age_cross = Tent("age", (0, 24, 60, 120), reference=24)
    design = Design([purpose, age, incentive, Cross(age_cross, incentive, drop=["age", "incentive"]),
                     Cross(purpose, age_cross, incentive, drop=["purpose", ("age", "incentive")])], intercept=True)
    Z = design.fit_transform(df)
    groups = design.groups()
    assert list(Z.columns) == [c for columns in groups.values() for c in columns]
    assert groups["intercept"] == ["intercept"] and (Z["intercept"] == 1).all()
    assert groups["age"] == ["age_at_0", "age_at_12", "age_at_60", "age_at_120"]
    assert groups["purpose"] == ["purpose_C", "purpose_N"]
    assert len(groups["age_x_incentive"]) == 3 * 4 and len(groups["purpose_x_age_x_incentive"]) == 2 * (3 + 4 + 3 * 4)
    assert rank(Z) == Z.shape[1]
    assert rank(Z, every_column(design, df)) == Z.shape[1]


@pytest.mark.parametrize("hierarchy", ["three mains and their triple, no pairs",
                                       "a curve surface per purpose, with age × incentive below it"])
def test_hierarchies_the_tensor_term_cannot_write(hierarchy):
    df = frame()
    purpose, age, incentive, ltv = encodings()
    if hierarchy.startswith("three"):
        terms = [age, incentive, ltv, Cross(age, incentive, ltv, drop=[("age", "incentive"), ("age", "ltv"),
                                                                       ("incentive", "ltv")])]
    else:
        terms = [purpose, age, incentive, Cross(age, incentive, drop=["age", "incentive"]),
                 Cross(purpose, age, incentive, drop=["purpose", ("age", "incentive")])]
    Z = Design(terms, intercept=True).fit_transform(df)
    tensor = Design([Cross(*terms[-1].encodings, drop=[])], intercept=False).fit_transform(df)
    assert rank(Z) == Z.shape[1]
    assert rank(Z, tensor) == Z.shape[1] == rank(tensor)               # the full tensor product's span


def test_conditions_that_others_imply_may_be_left_out():
    df = frame()
    purpose, age, incentive, _ = encodings()
    lower = [purpose, age, incentive, Cross(age, incentive, drop=["age", "incentive"])]
    short = Design([*lower, Cross(purpose, age, incentive, drop=["purpose", ("age", "incentive")])], intercept=True)
    every = Design([*lower, Cross(purpose, age, incentive, drop=[("purpose", "age", "incentive"), ("age", "incentive"),
                                                                   ("purpose", "incentive"), ("purpose", "age"),
                                                                   "purpose"])], intercept=True)
    pd.testing.assert_frame_equal(short.fit_transform(df), every.fit_transform(df))


@pytest.mark.parametrize("drop, problem", [
    (["age"], r"keeps the part that varies only with age, also provided by Tent\('age'\)"),
    ([("age", "incentive")], r"keeps the part that varies only with age.*only with incentive"),
    (["age", "incentive", "purpose"], r"drop names \['purpose'\]"),
])
def test_a_wrong_drop_names_each_effect_and_the_right_drop(drop, problem):
    _, age, incentive, _ = encodings()
    with pytest.raises(ValueError, match=problem) as error:
        Design([age, incentive, Cross(age, incentive, drop=drop)], intercept=True).check()
    if "drop names" not in problem:
        message = str(error.value)
        assert "the intercept: ('age', 'incentive')\n  Tent('age'): 'incentive'\n  Tent('incentive'): 'age'" in message
        assert "drop=['age', 'incentive']" in message


def test_a_wrong_three_way_drop_lists_the_lower_cross_too():
    purpose, age, incentive, _ = encodings()
    terms = [purpose, age, incentive, Cross(age, incentive, drop=["age", "incentive"]),
             Cross(purpose, age, incentive, drop=["purpose", "age", "incentive"])]
    with pytest.raises(ValueError, match="removes the part that varies with purpose and age together") as error:
        Design(terms, intercept=True).check()
    assert "Cross('age_x_incentive'): 'purpose'" in str(error.value)
    assert "drop=['purpose', ('age', 'incentive')]" in str(error.value)


def test_reference_drop_and_intercept_have_no_defaults():
    _, age, incentive, _ = encodings()
    with pytest.raises(TypeError, match="reference"):
        Tent("age", AGE_KNOTS)
    with pytest.raises(TypeError, match="reference"):
        OneHot("purpose")
    with pytest.raises(TypeError, match="drop"):
        Cross(age, incentive)
    with pytest.raises(TypeError, match="drop is a list"):
        Cross(age, incentive, drop=("age", "incentive"))     # a list of two conditions or one condition: write which
    with pytest.raises(TypeError, match="intercept"):
        Design([age, incentive])
    with pytest.raises(TypeError, match="intercept is True or False"):
        Design([age, incentive], intercept=None).check()


def test_without_an_intercept_one_term_keeps_all_its_columns():
    df = frame()
    single = Design([Tent("age", AGE_KNOTS, reference=None)], intercept=False)
    Z = single.fit_transform(df)
    assert list(Z.columns) == [f"age_at_{k}" for k in AGE_KNOTS] and rank(Z) == 5
    np.testing.assert_allclose(Z.sum(axis=1), 1)                        # the tent carries the constant

    purpose, age, incentive, _ = encodings()
    full_purpose = OneHot("purpose", reference=None)
    without = Design([full_purpose, age, incentive, Cross(age, incentive, drop=["age", "incentive"])],
                     intercept=False).fit_transform(df)
    with_intercept = Design([purpose, age, incentive, Cross(age, incentive, drop=["age", "incentive"])],
                            intercept=True).fit_transform(df)
    assert list(without.columns[:3]) == ["purpose_C", "purpose_N", "purpose_P"]
    assert without.shape == with_intercept.shape
    assert rank(without) == rank(with_intercept) == rank(without, with_intercept) == without.shape[1]


def test_a_cross_alone_without_an_intercept_is_the_full_tensor_product():
    df = frame()
    age = Tent("age", AGE_KNOTS, reference=None)
    incentive = Tent("incentive", INCENTIVE_KNOTS, reference=None)
    Z = Design([Cross(age, incentive, drop=[])], intercept=False).fit_transform(df)
    assert Z.shape[1] == 25 and rank(Z) == 25
    with pytest.raises(ValueError, match=r"drop names \['age'\], which have reference=None"):
        Cross(age, incentive, drop=["age"])
    with pytest.raises(ValueError, match="The constant is carried by the intercept and Cross"):
        Design([Cross(age, incentive, drop=[])], intercept=True).check()


@pytest.mark.parametrize("intercept, references, problem", [
    (True, (None, 24), r"carried by the intercept and OneHot\('purpose'\), so the design is collinear"),
    (False, (None, None), r"carried by OneHot\('purpose'\) and Tent\('age'\), so the design is collinear"),
    (False, ("P", 24), "carried by no term, so the design gains a constraint"),
])
def test_exactly_one_term_carries_the_constant(intercept, references, problem):
    terms = [OneHot("purpose", reference=references[0]), Tent("age", AGE_KNOTS, reference=references[1])]
    with pytest.raises(ValueError, match=problem):
        Design(terms, intercept=intercept).check()


def test_references_are_knots_or_levels():
    with pytest.raises(ValueError, match="reference 30 is not one of the knots"):
        Tent("age", AGE_KNOTS, reference=30)
    with pytest.raises(ValueError, match="reference 'X' is not one of the levels"):
        Design([OneHot("purpose", reference="X")], intercept=True).fit(frame())


def test_a_cross_may_use_coarser_knots_but_not_extra_ones():
    df = frame()
    _, age, incentive, _ = encodings()
    coarse = Tent("incentive", (-3, 0, 3), reference=0)
    Z = Design([age, incentive, Cross(age, coarse, drop=["age", "incentive"])], intercept=True).fit_transform(df)
    assert rank(Z) == Z.shape[1] == 1 + 4 + 4 + 4 * 2
    with pytest.raises(ValueError, match=r"incentive knots \[-3.0, -1.0, 0.0, 1.0, 3.0\] are not a subset"):
        Design([age, coarse, Cross(age, incentive, drop=["age", "incentive"])], intercept=True).check()


def test_a_cross_with_its_own_reference_spans_the_same_model_but_warns():
    df = frame()
    _, age, incentive, _ = encodings()
    same = Design([age, incentive, Cross(age, incentive, drop=["age", "incentive"])], intercept=True).fit_transform(df)
    age_60 = Tent("age", AGE_KNOTS, reference=60)
    with pytest.warns(UserWarning, match=r"age has different references \(24 in Tent\('age'\), 60 in Cross"):
        other = Design([age, incentive, Cross(age_60, incentive, drop=["age", "incentive"])], intercept=True)
        moved = other.fit_transform(df)
    kept = other.groups()["age_x_incentive"]
    assert "age_at_24_x_incentive_at_-3" in kept and "age_at_60_x_incentive_at_-3" not in kept
    assert rank(same) == rank(moved) == rank(same, moved) == same.shape[1]


def test_the_design_checks_its_terms_together():
    purpose, age, incentive, ltv = encodings()
    with pytest.raises(ValueError, match=r"share the names \['age'\]"):
        Design([age, Tent("age", (0, 60, 120), reference=60)], intercept=True).check()
    with pytest.raises(ValueError, match="Two crosses read the same columns"):
        Design([age, incentive, Cross(age, incentive, drop=["age", "incentive"]),
                Cross(incentive, age, drop=["incentive", "age"])], intercept=True).check()
    with pytest.raises(ValueError, match="age is encoded both as OneHot and as Tent"):
        Design([age, Cross(OneHot("age", reference=24), incentive, drop=[("age", "incentive")])],
               intercept=True).check()
    with pytest.raises(ValueError, match="no drop fits this design. the part that varies with age and incentive "
                                         "together is provided by Cross\\('age_x_incentive'\\), but no term provides "
                                         "the part that varies only with age"):
        Design([ltv, Cross(age, incentive, ltv, drop=["ltv"]), Cross(age, incentive, drop=["age", "incentive"])],
               intercept=True).check()


def test_fit_leaves_the_terms_unfitted_and_fits_a_shared_encoding_once():
    df = frame()
    purpose, age, _, _ = encodings()
    design = Design([purpose, age, Cross(purpose, age, drop=["purpose", "age"])], intercept=True).fit(df)
    assert not hasattr(purpose, "levels_") and not hasattr(age, "transformer_")
    main, _, fitted = design.terms_
    assert fitted.encodings[0] is main and main.levels_ == ["C", "N", "P"]


def test_transform_reads_new_rows_with_the_learned_levels():
    df = frame()
    purpose, age, incentive, _ = encodings()
    design = Design([purpose, age, incentive, Cross(purpose, age, drop=["purpose", "age"])], intercept=True)
    Z = design.fit_transform(df, dtype=np.float32)
    rows = design.transform(df.iloc[:7], dtype=np.float32)
    pd.testing.assert_frame_equal(rows, Z.iloc[:7])
    assert (rows.dtypes == np.float32).all()
    with pytest.raises(ValueError, match="levels outside"):
        design.transform(df.assign(purpose="R"))


def test_fit_transform_warns_about_cells_without_data():
    df = frame()
    df.loc[(df["purpose"] == "N") & (df["age"] > 24), "purpose"] = "C"
    purpose, age, _, _ = encodings()
    with pytest.warns(UserWarning, match=r"2 columns have no data.*'purpose_N_x_age_at_60', 'purpose_N_x_age_at_120'"):
        Design([purpose, age, Cross(purpose, age, drop=["purpose", "age"])], intercept=True).fit_transform(df)
