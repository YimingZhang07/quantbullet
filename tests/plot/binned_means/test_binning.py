"""BinSpec: edges, quantiles, steps and rounding, for pandas and Polars input."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import polars as pl
import pytest

from quantbullet.plot.binned_means import BinSpec, summarize_binned_means


def test_edges_include_lowest_and_keep_empty_bin():
    df = pd.DataFrame({"x": [-1., 0., 1., 3., 5., np.nan], "y": np.arange(6.)})
    data = summarize_binned_means(df, x="x", y="y", bins={"x": BinSpec.edges(np.array([0, 1, 2, 4]))})
    assert data.summary["count"].tolist() == [2, 0, 1]
    assert np.isnan(data.summary.iloc[1]["y__mean"])
    assert data.excluded_count == 3
    assert 0 in data.levels["x"][0]


def test_quantiles_are_global_not_per_facet():
    df = pd.DataFrame({"x": [0., 1., 2., 3., 10., 11., 12., 13.], "g": [0] * 4 + [1] * 4, "y": range(8)})
    data = summarize_binned_means(df, x="x", y="y", col="g", bins={"x": BinSpec.quantile(2)})
    assert data.bin_info["x"]["edges"] == (0., 6.5, 13.)
    assert data.summary.loc[data.summary["col"] == 0, "count"].tolist() == [4, 0]
    assert data.summary.loc[data.summary["col"] == 1, "count"].tolist() == [0, 4]


@pytest.mark.parametrize("spec", [None, BinSpec.edges([0, 0.5, 1, 2]), BinSpec.quantile(3), BinSpec.step(0.5)])
def test_pandas_polars_parity_for_each_binning_strategy(spec):
    records = {"x": [0., 0.2, 0.8, 1., 1.5, None], "g": [0, 0, 1, 1, 1, 1],
               "y": [1., None, 3., 4., 5., 6.], "z": [2., 4., None, 8., 10., 12.],
               "w": [1., 2., 0., None, 4., 1.]}
    kwargs = dict(x="x", y=["y", "z"], weight="w", group="g", bins={"x": spec} if spec else None)
    pandas_data = summarize_binned_means(pd.DataFrame(records), **kwargs)
    polars_data = summarize_binned_means(pl.DataFrame(records), **kwargs)
    assert_frame_equal(pandas_data.summary, polars_data.summary)
    assert pandas_data.bin_info == polars_data.bin_info
    assert pandas_data.excluded_count == polars_data.excluded_count


def test_round_ties_and_step_have_distinct_membership_and_input_parity():
    records = {'x': [-.75, -.25, .25, .75, 1., None], 'y': [1., 2., 3., 4., 5., 6.]}
    options = dict(x='x', y='y', bins={'x': BinSpec.round(.5)})
    a = summarize_binned_means(pd.DataFrame(records), **options)
    b = summarize_binned_means(pl.DataFrame(records), **options)
    assert_frame_equal(a.summary, b.summary)
    assert a.x_positions.tolist() == [-1., 0., 1.]
    assert a.summary['count'].tolist() == [1, 2, 2]
    expected = pl.DataFrame(records).select((pl.col('x')/.5).round()*.5)['x'].drop_nulls().to_list()
    assert sorted(expected) == [-1., 0., 0., 1., 1.]
    step = summarize_binned_means(pl.DataFrame(records), x='x', y='y', bins={'x': BinSpec.step(.5)})
    assert step.summary['count'].tolist() != a.summary['count'].tolist()


def test_decimal_rounding_uses_polars_arithmetic_for_both_inputs():
    records = {'x': [.15, .35, .95], 'y': [1., 2., 3.]}
    expected = pl.DataFrame(records).select((pl.col('x')/.1).round()*.1)['x'].to_list()
    for frame in [pd.DataFrame(records), pl.DataFrame(records)]:
        result = summarize_binned_means(frame, x='x', y='y', bins={'x':BinSpec.round(.1)})
        np.testing.assert_allclose(result.x_positions, expected)
        assert result.summary['count'].tolist() == [1,1,1]


def test_negative_zero_rounds_into_one_positive_level():
    df=pl.DataFrame({"x":[-.1,.1,-.6],"y":[1.,2.,3.]})
    data=summarize_binned_means(df,x="x",y="y",bins={"x":BinSpec.round(.5)})
    assert data.levels["x"]==(-.5,0.)
    assert np.signbit(data.x_positions).tolist()==[True,False]
    assert data.summary["count"].tolist()==[1,2]
