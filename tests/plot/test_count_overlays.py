"""Count overlays, bin-level transforms, preaggregated summaries and fast keys."""

from datetime import date
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytest

from quantbullet.plot.grouped_data import BinSpec, GroupedMeansData, summarize_grouped_means
from quantbullet.plot.grouped_means import draw_grouped_means, plot_grouped_means


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_summary_adapter_preserves_estimates_and_completes_facets():
    summary=pd.DataFrame({"bin":[1.,3.,1.,2.,3.],"facet":[0,0,1,1,1],
                          "n":[10,2,4,8,9],"actual":[.1,.3,.4,.5,.6],"pred":[.2,.4,.3,.5,.7]})
    before=summary.copy(deep=True)
    data=GroupedMeansData.from_summary(summary,x="bin",col="facet",count="n",
                                      mean_columns={"actual":"actual","pred":"pred"}).mask_support(5)
    assert data.summary["count"].sum()==33
    first=data.summary.loc[data.summary["col"]==0]
    assert first["count"].tolist()==[10,0,2]
    assert first["actual__mean"].tolist()[0]==.1
    assert first["actual__mean"].isna().tolist()==[False,True,True]
    result=draw_grouped_means(data,wrap=2,count_scale="shared")
    assert result.count_axes[0,0].get_ylim()==result.count_axes[0,1].get_ylim()
    assert sum(bar.get_height() for ax in result.count_axes.flat for bar in ax.patches)==33
    assert sum(len(ax.patches) for ax in result.count_axes.flat)==6  # not 12 for two metrics
    pd.testing.assert_frame_equal(before,summary)


def test_ordered_categories_keep_order_and_empty_levels():
    table=pd.DataFrame({"x":pd.Categorical(["B","A"],categories=["B","A","C"],ordered=True),
                        "count":[2,1],"y":[.2,.1]})
    data=GroupedMeansData.from_summary(table,x="x",mean_columns={"y":"y"})
    assert data.levels["x"]==("B","A","C")
    assert data.summary["count"].tolist()==[2,1,0]
    assert np.isnan(data.summary["y__mean"].iloc[2])


def test_transform_and_support_mask_apply_after_weighting_without_mutation():
    df=pl.DataFrame({"x":[1,1,2],"y":[0.,1.,0.],"p":[.1,.3,.2],"w":[1.,3.,2.]})
    data=summarize_grouped_means(df,x="x",y=["y","p"],weight="w")
    before=data.summary.copy(deep=True)
    cpr=lambda value: 1-(1-value)**12
    shown=data.map_means(cpr).mask_support(2)
    assert shown.summary["y__mean"].iloc[0]==pytest.approx(cpr(.75))  # CPR of weighted SMM
    assert shown.summary["p__mean"].iloc[0]==pytest.approx(cpr(.25))
    assert np.isnan(shown.summary["y__mean"].iloc[1]) and shown.summary["count"].tolist()==[2,1]
    assert shown.summary["y__weighted_sum"].tolist()==before["y__weighted_sum"].tolist()
    pd.testing.assert_frame_equal(data.summary,before)
    with pytest.raises(ValueError,match="nonnegative"):
        data.mask_support(-1)
    with pytest.raises(ValueError,match="unknown metrics"):
        data.map_means(cpr,metrics=["missing"])


@pytest.mark.parametrize("backend",["pandas","polars"])
def test_plot_counts_mask_and_fixed_markers(backend):
    records={"x":[-.75,-.6,-.5,-.4,-.2],"y":[0.,1.,0.,1.,0.],
             "p":[.1,.4,.2,.3,.2],"w":[1.,3.,2.,1.,1.]}
    source=pl.DataFrame(records) if backend=="polars" else pd.DataFrame(records)
    result=plot_grouped_means(source,x="x",y=["y","p"],weight="w",
                              bins={"x":BinSpec.round(.25)},min_count=2)
    np.testing.assert_allclose(result.axes[0,0].lines[0].get_xdata(),[-.75,-.5,-.25])
    np.testing.assert_allclose(result.axes[0,0].lines[0].get_ydata(),[np.nan,4/6,np.nan],equal_nan=True)
    assert [bar.get_height() for bar in result.count_axes[0,0].patches]==[1,3,1]
    assert result.axes[0,0].lines[0].get_markersize()==result.axes[0,0].lines[1].get_markersize()
    legends=[text.get_text() for legend in result.fig.legends for text in legend.get_texts()]
    assert "Count (right axis)" in legends and "Size" not in legends


def test_date_positions_preserve_calendar_gaps():
    df=pl.DataFrame({"x":[date(2025,1,1),date(2025,3,1)],"y":[.1,.2]})
    result=plot_grouped_means(df,x="x",y="y")
    positions=result.axes[0,0].lines[0].get_xdata()
    assert positions[1]-positions[0]==59  # true dates, not two categorical positions
    assert result.count_axes[0,0].get_ylabel()=="Count"


def test_existing_axes_legend_placement_and_empty_input():
    fig,axes=plt.subplots(1,2)
    data=GroupedMeansData.from_summary(pd.DataFrame({"x":[1,2],"count":[2,1],"y":[.1,.2]}),
                                      x="x",mean_columns={"y":"y"})
    plotted=draw_grouped_means(data,ax=axes[0])
    assert plotted.fig is fig and plotted.axes[0,0] is axes[0]
    assert not fig.legends and axes[0].get_legend() is not None
    draw_grouped_means(data,ax=axes[1],legend="none")
    assert axes[1].get_legend() is None
    with pytest.raises(ValueError,match="legend"):
        draw_grouped_means(data,legend="side")
    empty=pl.DataFrame({"x":[],"y":[]},schema={"x":pl.Float64,"y":pl.Float64})
    result=plot_grouped_means(empty,x="x",y="y",bins={"x":BinSpec.quantile(4)})
    assert result.summary.empty and result.axes.shape==(1,1)


def test_falsey_facets_keep_separate_counts():
    df=pl.DataFrame({"x":[1,2,1],"facet":[False,False,True],"y":[.1,.2,.8]})
    result=plot_grouped_means(df,x="x",y="y",col="facet",wrap=3,count_scale="shared")
    np.testing.assert_allclose(result.axes[0,0].lines[0].get_ydata(),[.1,.2])
    np.testing.assert_allclose(result.axes[0,1].lines[0].get_ydata(),[.8,np.nan],equal_nan=True)
    assert [sum(p.get_height() for p in ax.patches) for ax in result.count_axes.flat]==[2,1]


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
    data=summarize_grouped_means(df,x="x",y="y",col="g",bins={"x":BinSpec.round(.5)})
    assert data.summary["count"].sum()==n
    assert sizes and max(sizes)<100


def test_negative_zero_rounds_into_one_positive_level():
    df=pl.DataFrame({"x":[-.1,.1,-.6],"y":[1.,2.,3.]})
    data=summarize_grouped_means(df,x="x",y="y",bins={"x":BinSpec.round(.5)})
    assert data.levels["x"]==(-.5,0.)
    assert np.signbit(data.x_positions).tolist()==[True,False]
    assert data.summary["count"].tolist()==[1,2]


def test_declared_category_orders_match_for_pandas_and_polars():
    categories=["B","A","C"]
    frames=[pd.DataFrame({"x":pd.Categorical(["B","A",None],categories=categories,ordered=True),"y":[1.,2.,3.]}),
            pl.DataFrame({"x":pl.Series(["B","A",None],dtype=pl.Enum(categories)),"y":[1.,2.,3.]})]
    for frame in frames:
        data=summarize_grouped_means(frame,x="x",y="y")
        assert data.levels["x"]==("B","A","C")
        assert data.summary["count"].tolist()==[1,1,0] and data.excluded_count==1


def test_pandas_nullable_integer_keys_keep_integer_levels():
    df=pd.DataFrame({"age":pd.array([1,2,None,2],dtype="Int64"),"y":[1.,2.,3.,4.]})
    data=summarize_grouped_means(df,x="age",y="y")
    assert data.levels["x"]==(1,2) and data.excluded_count==1
    assert data.summary["y__mean"].tolist()==[1.,3.]


def test_legacy_entry_point_is_deprecated_but_keeps_point_sizes():
    from quantbullet.plot.binned_plots import plot_binned_actual_vs_pred
    df=pd.DataFrame({"x":[1.,2.,3.,4.],"y":[.1,.2,.3,.4],"p":[.1,.2,.3,.4]})
    with pytest.warns(DeprecationWarning,match="plot_grouped_means"):
        fig,_=plot_binned_actual_vs_pred(df,"x","y","p",bins="discrete")
    assert any(legend.get_title().get_text()=="Size" for legend in fig.legends)


@pytest.mark.parametrize("loss",["poisson","mse"])
def test_implied_actual_formula_and_cache_unchanged(loss):
    from quantbullet.linear_product_model import LinearProductModelToolkit
    from quantbullet.model.feature import DataType,Feature,FeatureRole,FeatureSpec
    from types import SimpleNamespace
    from unittest.mock import patch
    toolkit=LinearProductModelToolkit(FeatureSpec([
        Feature("x",DataType.FLOAT,FeatureRole.MODEL_INPUT),Feature("y",DataType.FLOAT,FeatureRole.TARGET)]))
    raw={"x":pd.DataFrame({"feature_value":[1,1,2],"y":[0.,1.,0.],
                           "m":[.2,.4,.3],"model_pred":[1.,2.,1.5],"w":[1.,3.,1.]})}
    reference=toolkit._aggregate_implied_data(raw,{"x":"discrete"},100,loss=loss)["x"]
    with patch.object(toolkit,"compute_implied_actual_data",return_value=raw):
        fig,axes=toolkit.plot_implied_actuals(SimpleNamespace(loss_=loss),None,bin_config={"x":"discrete"},min_count=2,n_cols=1)
    actual=toolkit.implied_actual_data_caches["x"].agg_df
    pd.testing.assert_frame_equal(actual,reference)
    expected=reference["implied_actual"].to_numpy().copy();expected[reference["count"].to_numpy()<2]=np.nan
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(),expected,equal_nan=True)
    assert [bar.get_height() for bar in toolkit.implied_actual_count_axes[0].patches]==[2,1]
    assert axes[0].get_legend() is not None
