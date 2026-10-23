"""TwoWayDiagnostic: reading, excluding and checking the five input columns."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from quantbullet.diagnostics import TwoWayDiagnostic

COLUMNS = dict(x="x", z="z", y="y", pred="p", weight="w")


def raw():
    return {
        "x": [1., 2., 3., 0., 5., 6.],
        "z": [1., 2., 3., 4., 5., np.inf],
        "y": [0., 1., np.nan, 0., 1., 0.],
        "p": [.1, .2, .3, .4, .5, .6],
        "w": [1., 2., 1., 1., 1., 1.],
    }


def test_pandas_and_polars_read_the_same_rows_and_count_exclusions():
    from_pandas = TwoWayDiagnostic(pd.DataFrame(raw()), **COLUMNS, x_scale="log")
    from_polars = TwoWayDiagnostic(pl.DataFrame(raw()), **COLUMNS, x_scale="log")
    for diag in (from_pandas, from_polars):
        assert diag.excluded == {"missing": 2, "nonpositive_x": 1}
        assert diag.n_obs == 3 and diag.input_rows == 6
        assert diag._array("x").tolist() == [1., 2., 5.]
        assert diag.x_range == (1., 5.) and diag.z_range == (1., 5.)
    assert np.array_equal(from_pandas._array("weight"), from_polars._array("weight"))


def test_unweighted_rows_get_unit_weights_and_nulls_are_missing():
    frame = pl.DataFrame(raw()).with_columns(pl.when(pl.col("x") == 2).then(None).otherwise(pl.col("y")).alias("y"))
    diag = TwoWayDiagnostic(frame, x="x", z="z", y="y", pred="p")
    assert diag.excluded == {"missing": 3}
    assert diag._array("weight").tolist() == [1.] * diag.n_obs


@pytest.mark.parametrize("change, error, match", [
    (dict(pred="missing"), ValueError, "missing columns"),
    (dict(pred="x"), ValueError, "distinct"),
    (dict(x_scale="logit"), ValueError, "x_scale"),
    (dict(y="label"), TypeError, "numeric"),
])
def test_bad_arguments_raise(change, error, match):
    frame = pd.DataFrame(raw()).assign(label=list("abcdef"))
    with pytest.raises(error, match=match):
        TwoWayDiagnostic(frame, **{**COLUMNS, **change})


def test_the_weight_may_reuse_the_x_column():
    diag = TwoWayDiagnostic(pd.DataFrame(raw()), x="x", z="z", y="y", pred="p", weight="x")
    assert np.array_equal(diag._array("weight"), diag._array("x"))


def test_negative_weights_and_empty_inputs_raise():
    with pytest.raises(ValueError, match="nonnegative"):
        TwoWayDiagnostic(pd.DataFrame({**raw(), "w": [1., -1., 1., 1., 1., 1.]}), **COLUMNS)
    with pytest.raises(ValueError, match="no rows"):
        TwoWayDiagnostic(pd.DataFrame({**raw(), "y": [np.nan] * 6}), **COLUMNS)
    with pytest.raises(TypeError, match="DataFrame"):
        TwoWayDiagnostic(raw(), **COLUMNS)
