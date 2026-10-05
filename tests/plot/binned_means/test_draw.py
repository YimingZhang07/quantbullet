"""Drawing: counts, legends, dates, facets, and per-axis scales, tick labels and titles."""
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytest

from quantbullet.plot.binned_means import (BinnedMeans, BinSpec, draw_binned_means, plot_binned_means,
                                           summarize_binned_means)
from tests.plot.helpers import LEVELS, facet_frame, visible_tick_labels


def test_falsey_facets_do_not_mix_and_count_not_duplicated():
    df = pd.DataFrame({"x": [1, 2, 1, 2], "a": [10., 20., 30., 40.],
                       "b": [11., 21., 31., 41.], "g": [0, 0, 1, 1]})
    result = plot_binned_means(df, x="x", y=["a", "b"], col="g")
    np.testing.assert_array_equal(result.axes[0, 0].lines[0].get_ydata(), [10, 20])
    np.testing.assert_array_equal(result.axes[0, 1].lines[0].get_ydata(), [30, 40])
    assert sum(p.get_height() for ax in result.bar_axes.flat for p in ax.patches) == 4
    assert "g: 0" in result.axes[0, 0].get_title()


def test_group_row_and_col_combine():
    df = pd.MultiIndex.from_product([[0, 1]] * 4, names=["r", "c", "g", "x"]).to_frame(index=False)
    df["y"] = np.arange(16.)
    result = plot_binned_means(df, x="x", y="y", group="g", row="r", col="c", bar_mode="stacked")
    assert result.axes.shape == (2, 2)
    for (r, c), ax in np.ndenumerate(result.axes):
        for g, line in enumerate(ax.lines):
            expected = df.loc[(df["r"] == r) & (df["c"] == c) & (df["g"] == g), "y"]
            np.testing.assert_array_equal(line.get_ydata(), expected)
    assert sum(p.get_height() for ax in result.bar_axes.flat for p in ax.patches) == 16
    assert result.axes[1, 0].get_title() == "r: 1 | c: 0"


def test_pad_to_wrap_keeps_every_wrap_column():
    data = summarize_binned_means(facet_frame(), x="x", y="y", col="f", bins={"x": BinSpec.step(.25)}).select(
        "col", ["a", "b"])
    assert draw_binned_means(data, wrap=3).axes.shape == (1, 2)
    padded = draw_binned_means(data, wrap=3, pad_to_wrap=True)
    assert padded.axes.shape == (1, 3) and not padded.axes[0, 2].get_visible()


@pytest.mark.parametrize("backend",["pandas","polars"])
def test_plot_counts_mask_and_fixed_markers(backend):
    records={"x":[-.75,-.6,-.5,-.4,-.2],"y":[0.,1.,0.,1.,0.],
             "p":[.1,.4,.2,.3,.2],"w":[1.,3.,2.,1.,1.]}
    source=pl.DataFrame(records) if backend=="polars" else pd.DataFrame(records)
    result=plot_binned_means(source,x="x",y=["y","p"],weight="w",
                              bins={"x":BinSpec.round(.25)},min_count=2)
    np.testing.assert_allclose(result.axes[0,0].lines[0].get_xdata(),[-.75,-.5,-.25])
    np.testing.assert_allclose(result.axes[0,0].lines[0].get_ydata(),[np.nan,4/6,np.nan],equal_nan=True)
    assert [bar.get_height() for bar in result.bar_axes[0,0].patches]==[1,3,1]
    assert result.axes[0,0].lines[0].get_markersize()==result.axes[0,0].lines[1].get_markersize()
    legends=[text.get_text() for legend in result.fig.legends for text in legend.get_texts()]
    assert "Count (right axis)" in legends and "Size" not in legends


def test_date_positions_preserve_calendar_gaps():
    df=pl.DataFrame({"x":[date(2025,1,1),date(2025,3,1)],"y":[.1,.2]})
    result=plot_binned_means(df,x="x",y="y")
    positions=result.axes[0,0].lines[0].get_xdata()
    assert positions[1]-positions[0]==59  # true dates, not two categorical positions
    assert result.bar_axes[0,0].get_ylabel()=="Count"


def test_existing_axes_legend_placement_and_empty_input():
    fig,axes=plt.subplots(1,2)
    data=BinnedMeans.from_summary(pd.DataFrame({"x":[1,2],"count":[2,1],"y":[.1,.2]}),
                                      x="x",mean_columns={"y":"y"})
    plotted=draw_binned_means(data,ax=axes[0])
    assert plotted.fig is fig and plotted.axes[0,0] is axes[0]
    assert not fig.legends and axes[0].get_legend() is not None
    draw_binned_means(data,ax=axes[1],legend="none")
    assert axes[1].get_legend() is None
    with pytest.raises(ValueError,match="legend"):
        draw_binned_means(data,legend="side")
    empty=pl.DataFrame({"x":[],"y":[]},schema={"x":pl.Float64,"y":pl.Float64})
    result=plot_binned_means(empty,x="x",y="y",bins={"x":BinSpec.quantile(4)})
    assert result.summary.empty and result.axes.shape==(1,1)


def test_falsey_facets_keep_separate_counts():
    df=pl.DataFrame({"x":[1,2,1],"facet":[False,False,True],"y":[.1,.2,.8]})
    result=plot_binned_means(df,x="x",y="y",col="facet",wrap=3,bar_scale="shared")
    np.testing.assert_allclose(result.axes[0,0].lines[0].get_ydata(),[.1,.2])
    np.testing.assert_allclose(result.axes[0,1].lines[0].get_ydata(),[.8,np.nan],equal_nan=True)
    assert [sum(p.get_height() for p in ax.patches) for ax in result.bar_axes.flat]==[2,1]


def test_external_axes_title_stays_on_the_curve_axes_with_shared_counts():
    data = summarize_binned_means(facet_frame(), x="x", y=["y", "p"], bins={"x": BinSpec.step(.25)})
    _, ax = plt.subplots()
    result = draw_binned_means(data, ax=ax, bar_scale="shared", title="Panel")
    assert ax.get_title() == "Panel"
    assert result.bar_axes[0, 0].get_title() == ""


def test_outer_titles_and_fixed_count_scale():
    data = summarize_binned_means(facet_frame(), x="x", y=["y", "p"], col="f", bins={"x": BinSpec.step(.25)})
    result = draw_binned_means(data, wrap=3, pad_to_wrap=True, bar_scale=2e5, y_titles="outer",
                                bar_titles="outer", x_titles="outer", y_ticks="outer",
                                bar_ticks="outer", y_label="Rate")
    axes, twins = result.axes, result.bar_axes
    assert [ax.get_ylabel() for ax in axes[0]] == ["Rate", "", ""]
    assert [twin.get_ylabel() for twin in twins[0]] == ["", "", "Count"]
    assert not axes[1, 2].get_visible() and twins[1, 1].get_ylabel() == "Count"
    # Only panels with a visible panel below drop the x label.
    assert [ax.get_xlabel() for ax in axes[0]] == ["", "", "x"]
    assert all(twin.get_ylim() == (0, 2e5) for twin in twins.flat if twin is not None)
    assert "200K" in visible_tick_labels(result.fig, twins[0, 2].yaxis)
    assert visible_tick_labels(result.fig, twins[0, 0].yaxis) == []


@pytest.mark.parametrize("y_scale,y_ticks,ticks", [
    ("free", "all", [True, True, True]),
    ("shared", "all", [True, True, True]),  # a shared scale alone keeps every tick label
    ("shared", "outer", [True, False, False]),
    ((0, .2), "all", [True, True, True]),
    ((0, .2), "outer", [True, False, False]),
])
def test_primary_scale_and_tick_labels_are_separate(y_scale, y_ticks, ticks):
    data = summarize_binned_means(facet_frame(), x="x", y=["y", "p"], col="f", bins={"x": BinSpec.step(.25)})
    result = draw_binned_means(data, wrap=3, pad_to_wrap=True, y_scale=y_scale, y_ticks=y_ticks)
    row = result.axes[0]
    assert [bool(visible_tick_labels(result.fig, ax.yaxis)) for ax in row] == ticks
    assert (len({ax.get_ylim() for ax in row}) == 1) == (y_scale != "free")


@pytest.mark.parametrize("y_ticks,bar_ticks", [("outer", "all"), ("all", "outer")])
def test_each_y_axis_sets_its_own_tick_labels(y_ticks, bar_ticks):
    data = summarize_binned_means(facet_frame(), x="x", y=["y", "p"], col="f", bins={"x": BinSpec.step(.25)})
    result = draw_binned_means(data, wrap=3, pad_to_wrap=True, y_scale="shared", bar_scale="shared",
                                y_ticks=y_ticks, bar_ticks=bar_ticks)
    primary = [bool(visible_tick_labels(result.fig, ax.yaxis)) for ax in result.axes[0]]
    counts = [bool(visible_tick_labels(result.fig, twin.yaxis)) for twin in result.bar_axes[0]]
    assert primary == ([True, False, False] if y_ticks == "outer" else [True] * 3)
    assert counts == ([False, False, True] if bar_ticks == "outer" else [True] * 3)


def test_axis_titles_and_tick_labels_are_independent():
    data = summarize_binned_means(facet_frame(), x="x", y=["y", "p"], col="f", bins={"x": BinSpec.step(.25)})
    shared = dict(wrap=3, pad_to_wrap=True, y_scale="shared", bar_scale="shared", y_label="Rate")
    titles_only = draw_binned_means(data, y_titles="outer", bar_titles="outer", x_titles="outer", **shared)
    axes, twins = titles_only.axes[0], titles_only.bar_axes[0]
    assert [ax.get_ylabel() for ax in axes] == ["Rate", "", ""]
    assert [twin.get_ylabel() for twin in twins] == ["", "", "Count"]
    assert all(visible_tick_labels(titles_only.fig, ax.yaxis) for ax in [*axes, *twins])
    ticks_only = draw_binned_means(data, y_ticks="outer", bar_ticks="outer", **shared)
    axes, twins = ticks_only.axes[0], ticks_only.bar_axes[0]
    assert [ax.get_ylabel() for ax in axes] == ["Rate"] * 3
    assert [bool(visible_tick_labels(ticks_only.fig, ax.yaxis)) for ax in axes] == [True, False, False]
    assert [bool(visible_tick_labels(ticks_only.fig, twin.yaxis)) for twin in twins] == [False, False, True]


@pytest.mark.parametrize("titles", [
    dict(y_titles="outer"), dict(bar_titles="outer"), dict(x_titles="outer"),
    dict(y_titles="outer", x_titles="outer"),
])
def test_each_axis_sets_its_own_titles(titles):
    data = summarize_binned_means(facet_frame(), x="x", y=["y", "p"], col="f", bins={"x": BinSpec.step(.25)})
    result = draw_binned_means(data, wrap=3, pad_to_wrap=True, y_label="Rate", **titles)
    axes, twins = result.axes[0], result.bar_axes[0]
    outer = {name: value == "outer" for name, value in titles.items()}
    assert [ax.get_ylabel() for ax in axes] == (["Rate", "", ""] if outer.get("y_titles") else ["Rate"] * 3)
    assert [twin.get_ylabel() for twin in twins] == (["", "", "Count"] if outer.get("bar_titles") else ["Count"] * 3)
    # Columns with a visible panel below drop the x title only under x_titles='outer'.
    assert [ax.get_xlabel() for ax in axes] == (["", "", "x"] if outer.get("x_titles") else ["x"] * 3)


def test_outer_tick_labels_need_a_common_scale_and_valid_values():
    data = summarize_binned_means(facet_frame(), x="x", y=["y", "p"], col="f", bins={"x": BinSpec.step(.25)})
    with pytest.raises(ValueError, match="y_ticks='outer' needs"):
        draw_binned_means(data, wrap=3, y_scale="free", y_ticks="outer")
    with pytest.raises(ValueError, match="bar_ticks='outer' needs"):
        draw_binned_means(data, wrap=3, bar_scale="free", bar_ticks="outer")
    for bad in (dict(y_ticks=True), dict(bar_ticks="inner"), dict(x_titles="none")):
        with pytest.raises(ValueError, match="must be 'all' or 'outer'"):
            draw_binned_means(data, wrap=3, **bad)
    for bad in ("same", True, 0, -1., float("nan")):
        with pytest.raises(ValueError, match="bar_scale"):
            draw_binned_means(data, wrap=3, bar_scale=bad)


def test_fixed_primary_range_applies_to_every_panel():
    data = summarize_binned_means(facet_frame(), x="x", y=["y", "p"], col="f", bins={"x": BinSpec.step(.25)})
    result = draw_binned_means(data, wrap=3, pad_to_wrap=True, y_scale=(-.01, .2))
    assert [ax.get_ylim() for ax in result.axes.flat[:len(LEVELS)]] == [(-.01, .2)] * len(LEVELS)
    for bad in ((.2, .1), (0, float("inf")), (0,), "same", True):
        with pytest.raises(ValueError, match="y_scale"):
            draw_binned_means(data, wrap=3, y_scale=bad)


@pytest.mark.parametrize("bar_mode", ["total", "stacked"])
def test_weight_bars_sum_the_weight_and_take_its_label(bar_mode):
    df = pl.DataFrame({"x": [1, 1, 2, 2], "g": ["a", "b", "a", "b"],
                       "y": [.1, .2, .3, None], "w": [100., 50., 200., 25.]})
    result = plot_binned_means(df, x="x", y="y", weight="w", group="g", bar_mode=bar_mode,
                               bar_value="weight", labels={"w": "UPB"})
    bars = result.bar_axes[0, 0]
    heights = [bar.get_height() for bar in bars.patches]
    # The row with a missing y still adds its weight.
    assert heights == ([150., 225.] if bar_mode == "total" else [100., 200., 50., 25.])
    assert bars.get_ylabel() == "UPB"
    legend = [text.get_text() for text in result.fig.legends[0].get_texts()]
    assert ("UPB (right axis)" in legend) if bar_mode == "total" else ("UPB: a (right axis)" in legend)
    counted = plot_binned_means(df, x="x", y="y", weight="w", bar_label="Loan-months")
    assert [bar.get_height() for bar in counted.bar_axes[0, 0].patches] == [2, 2]
    assert counted.bar_axes[0, 0].get_ylabel() == "Loan-months"


def test_weight_bars_need_a_weight_and_a_known_value():
    df = pl.DataFrame({"x": [1, 2], "y": [.1, .2], "w": [1., 2.]})
    with pytest.raises(ValueError, match="needs a weighted summary"):
        plot_binned_means(df, x="x", y="y", bar_value="weight")
    with pytest.raises(ValueError, match="bar_value must be"):
        plot_binned_means(df, x="x", y="y", weight="w", bar_value="upb")
    unlabeled = plot_binned_means(df, x="x", y="y", weight="w", bar_value="weight")
    assert unlabeled.bar_axes[0, 0].get_ylabel() == "w"  # the column name without a label
