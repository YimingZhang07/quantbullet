"""BinnedMeans: weighted statistics, level order, transforms, masks and subsets."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from quantbullet.plot.binned_means import BinnedMeans, BinSpec, draw_binned_means, summarize_binned_means
from tests.plot.helpers import LEVELS, facet_frame


def test_weighted_means_use_metric_specific_valid_rows():
    df = pd.DataFrame({"x": [0] * 5, "a": [1, 3, np.nan, 5, 7],
                       "b": [2, np.nan, 4, 8, 10], "w": [1, 3, 2, np.nan, 0]})
    before = df.copy(deep=True)
    data = summarize_binned_means(df, x="x", y=["a", "b"], weight="w")
    row = data.summary.iloc[0]
    assert row["count"] == 5
    assert row["a__mean"] == pytest.approx(2.5)
    assert row["b__mean"] == pytest.approx(10 / 3)
    assert row["a__valid_count"] == 3  # includes the zero-weight row
    assert row["a__weight_sum"] == 4
    assert row["b__weight_sum"] == 3
    pd.testing.assert_frame_equal(df, before)


def test_summary_adapter_preserves_estimates_and_completes_facets():
    summary=pd.DataFrame({"bin":[1.,3.,1.,2.,3.],"facet":[0,0,1,1,1],
                          "n":[10,2,4,8,9],"actual":[.1,.3,.4,.5,.6],"pred":[.2,.4,.3,.5,.7]})
    before=summary.copy(deep=True)
    data=BinnedMeans.from_summary(summary,x="bin",col="facet",count="n",
                                      mean_columns={"actual":"actual","pred":"pred"}).mask_sparse(5)
    assert data.summary["count"].sum()==33
    first=data.summary.loc[data.summary["col"]==0]
    assert first["count"].tolist()==[10,0,2]
    assert first["actual__mean"].tolist()[0]==.1
    assert first["actual__mean"].isna().tolist()==[False,True,True]
    result=draw_binned_means(data,wrap=2,count_scale="shared")
    assert result.count_axes[0,0].get_ylim()==result.count_axes[0,1].get_ylim()
    assert sum(bar.get_height() for ax in result.count_axes.flat for bar in ax.patches)==33
    assert sum(len(ax.patches) for ax in result.count_axes.flat)==6  # not 12 for two metrics
    pd.testing.assert_frame_equal(before,summary)


def test_ordered_categories_keep_order_and_empty_levels():
    table=pd.DataFrame({"x":pd.Categorical(["B","A"],categories=["B","A","C"],ordered=True),
                        "count":[2,1],"y":[.2,.1]})
    data=BinnedMeans.from_summary(table,x="x",mean_columns={"y":"y"})
    assert data.levels["x"]==("B","A","C")
    assert data.summary["count"].tolist()==[2,1,0]
    assert np.isnan(data.summary["y__mean"].iloc[2])


def test_transform_and_sparse_mask_apply_after_weighting_without_mutation():
    df=pl.DataFrame({"x":[1,1,2],"y":[0.,1.,0.],"p":[.1,.3,.2],"w":[1.,3.,2.]})
    data=summarize_binned_means(df,x="x",y=["y","p"],weight="w")
    before=data.summary.copy(deep=True)
    cpr=lambda value: 1-(1-value)**12
    shown=data.transform_means(cpr).mask_sparse(2)
    assert shown.summary["y__mean"].iloc[0]==pytest.approx(cpr(.75))  # CPR of weighted SMM
    assert shown.summary["p__mean"].iloc[0]==pytest.approx(cpr(.25))
    assert np.isnan(shown.summary["y__mean"].iloc[1]) and shown.summary["count"].tolist()==[2,1]
    assert shown.summary["y__weighted_sum"].tolist()==before["y__weighted_sum"].tolist()
    pd.testing.assert_frame_equal(data.summary,before)
    with pytest.raises(ValueError,match="nonnegative"):
        data.mask_sparse(-1)
    with pytest.raises(ValueError,match="unknown metrics"):
        data.transform_means(cpr,metrics=["missing"])


def test_polars_rows_are_not_factorized_in_pandas(monkeypatch):
    """Row-level keys stay in Polars; pandas only sees the aggregated table."""
    def forbidden(*args,**kwargs):
        raise AssertionError("row-level to_pandas is not expected")
    monkeypatch.setattr(pl.DataFrame,"to_pandas",forbidden)
    sizes=[]
    factorize=pd.factorize
    def counting(values,*args,**kwargs):
        sizes.append(len(values))
        return factorize(values,*args,**kwargs)
    monkeypatch.setattr(pd,"factorize",counting)
    rng=np.random.default_rng(1)
    n=5000
    df=pl.DataFrame({"x":rng.normal(size=n),"g":rng.choice(["a","b"],n),"y":rng.random(n)})
    data=summarize_binned_means(df,x="x",y="y",col="g",bins={"x":BinSpec.round(.5)})
    assert data.summary["count"].sum()==n
    assert sizes and max(sizes)<100


def test_declared_category_orders_match_for_pandas_and_polars():
    categories=["B","A","C"]
    frames=[pd.DataFrame({"x":pd.Categorical(["B","A",None],categories=categories,ordered=True),"y":[1.,2.,3.]}),
            pl.DataFrame({"x":pl.Series(["B","A",None],dtype=pl.Enum(categories)),"y":[1.,2.,3.]})]
    for frame in frames:
        data=summarize_binned_means(frame,x="x",y="y")
        assert data.levels["x"]==("B","A","C")
        assert data.summary["count"].tolist()==[1,1,0] and data.excluded_count==1


def test_pandas_nullable_integer_keys_keep_integer_levels():
    df=pd.DataFrame({"age":pd.array([1,2,None,2],dtype="Int64"),"y":[1.,2.,3.,4.]})
    data=summarize_binned_means(df,x="age",y="y")
    assert data.levels["x"]==(1,2) and data.excluded_count==1
    assert data.summary["y__mean"].tolist()==[1.,3.]


def test_select_limits_facets_and_keeps_full_level_metadata():
    data = summarize_binned_means(facet_frame(), x="x", y="y", col="f", bins={"x": BinSpec.step(.25)})
    selected = data.select("col", ["d", "b"])
    assert selected.levels["col"] == ("d", "b")
    assert set(selected.summary["col"]) == {"d", "b"}
    assert selected.bin_info["f"]["levels"] == tuple(LEVELS)
    with pytest.raises(ValueError, match="unknown"):
        data.select("col", ["z"])
    with pytest.raises(ValueError):
        data.select("x", [0])
