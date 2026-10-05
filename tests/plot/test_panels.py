"""Panel sets, outer labels, shared count scales and print formatting for report layouts."""
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytest

from quantbullet.linear_product_model.mortgage_diagnostics import MortgageColnames, MortgageDiagnostics
from quantbullet.plot.formatter import StepPercentFormatter, compact_number
from quantbullet.plot.grouped_data import BinSpec, summarize_grouped_means
from quantbullet.plot.grouped_means import draw_grouped_means
from quantbullet.plot.panels import PanelSet, panel_grid
from quantbullet.plot.theme import PRINT_THEME

LEVELS = ["a", "b", "c", "d", "e"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def facet_frame(n=600):
    rng = np.random.default_rng(0)
    return pl.DataFrame({"x": rng.uniform(0, 1, n), "y": rng.random(n) * .1, "p": rng.random(n) * .1,
                         "f": rng.choice(LEVELS, n)}).with_columns(pl.col("f").cast(pl.Enum(LEVELS)))


def visible_tick_labels(fig, axis):
    fig.canvas.draw()
    return [label.get_text() for label in axis.get_ticklabels() if label.get_visible()]


def test_compact_numbers_and_percent_decimals_follow_the_tick_step():
    assert [compact_number(v) for v in (0, 999, 1500, 450000, 1.2e6, 700000.0000001)] == \
        ["0", "999", "1.5K", "450K", "1.2M", "700K"]
    assert StepPercentFormatter().format_ticks([.01, .015, .02]) == ["1.0%", "1.5%", "2.0%"]
    assert StepPercentFormatter().format_ticks([.02, .04, .06]) == ["2%", "4%", "6%"]


def test_panel_set_draws_subsets_at_the_requested_size():
    calls = []

    def render(panels, n_cols, panel_size):
        calls.append((panels, n_cols, panel_size))
        return panel_grid(len(panels), n_cols, panel_size)[0]

    panels = PanelSet(["a", "b", "c", "d"], render, n_cols=3)
    assert panels.n_rows == 2
    fig = panels.draw(["a"], panel_size=(2, 1))
    assert tuple(fig.get_size_inches()) == (6, 1)  # the grid keeps its column width
    assert [ax.get_visible() for ax in fig.axes] == [True, False, False]
    wide = panels.subset(["d"], n_cols=1)
    assert tuple(wide.draw(panel_size=(4, 2)).get_size_inches()) == (4, 2)
    assert calls[-1] == (("d",), 1, (4, 2))
    with pytest.raises(ValueError, match="unknown"):
        panels.subset(["z"])
    with pytest.raises(ValueError):
        PanelSet([], render)


def test_select_limits_facets_and_keeps_full_level_metadata():
    data = summarize_grouped_means(facet_frame(), x="x", y="y", col="f", bins={"x": BinSpec.step(.25)})
    selected = data.select("col", ["d", "b"])
    assert selected.levels["col"] == ("d", "b")
    assert set(selected.summary["col"]) == {"d", "b"}
    assert selected.bin_info["f"]["levels"] == tuple(LEVELS)
    with pytest.raises(ValueError, match="unknown"):
        data.select("col", ["z"])
    with pytest.raises(ValueError):
        data.select("x", [0])


def test_external_axes_title_stays_on_the_curve_axes_with_shared_counts():
    data = summarize_grouped_means(facet_frame(), x="x", y=["y", "p"], bins={"x": BinSpec.step(.25)})
    _, ax = plt.subplots()
    result = draw_grouped_means(data, ax=ax, share_count_y=True, title="Panel")
    assert ax.get_title() == "Panel"
    assert result.count_axes[0, 0].get_title() == ""


def test_outer_labels_and_fixed_count_scale():
    data = summarize_grouped_means(facet_frame(), x="x", y=["y", "p"], col="f", bins={"x": BinSpec.step(.25)})
    result = draw_grouped_means(data, wrap=3, compact_cols=False, count_ylim=2e5, outer_labels=True, outer_ticks=True,
                                ylabel="Rate")
    axes, twins = result.axes, result.count_axes
    assert [ax.get_ylabel() for ax in axes[0]] == ["Rate", "", ""]
    assert [twin.get_ylabel() for twin in twins[0]] == ["", "", "Count"]
    assert not axes[1, 2].get_visible() and twins[1, 1].get_ylabel() == "Count"
    # Only panels with a visible panel below drop the x label.
    assert [ax.get_xlabel() for ax in axes[0]] == ["", "", "x"]
    assert all(twin.get_ylim() == (0, 2e5) for twin in twins.flat if twin is not None)
    assert "200K" in visible_tick_labels(result.fig, twins[0, 2].yaxis)
    assert visible_tick_labels(result.fig, twins[0, 0].yaxis) == []


@pytest.mark.parametrize("scale,outer_ticks,ticks", [
    (dict(share_y=False), False, [True, True, True]),
    (dict(share_y=False), True, [True, True, True]),  # independent scales always keep their ticks
    (dict(share_y=True), False, [True, True, True]),  # sharing alone only sets the scale
    (dict(share_y=True), True, [True, False, False]),
    (dict(share_y=False, ylim=(0, .2)), False, [True, True, True]),
    (dict(share_y=False, ylim=(0, .2)), True, [True, False, False]),
])
def test_primary_scale_and_tick_display_are_separate(scale, outer_ticks, ticks):
    data = summarize_grouped_means(facet_frame(), x="x", y=["y", "p"], col="f", bins={"x": BinSpec.step(.25)})
    result = draw_grouped_means(data, wrap=3, compact_cols=False, outer_ticks=outer_ticks, **scale)
    row = result.axes[0]
    assert [bool(visible_tick_labels(result.fig, ax.yaxis)) for ax in row] == ticks
    shared = scale["share_y"] or "ylim" in scale
    assert (len({ax.get_ylim() for ax in row}) == 1) == shared


def test_axis_titles_and_tick_labels_are_independent():
    data = summarize_grouped_means(facet_frame(), x="x", y=["y", "p"], col="f", bins={"x": BinSpec.step(.25)})
    shared = dict(wrap=3, compact_cols=False, share_y=True, share_count_y=True, ylabel="Rate")
    titles_only = draw_grouped_means(data, outer_labels=True, outer_ticks=False, **shared)
    axes, twins = titles_only.axes[0], titles_only.count_axes[0]
    assert [ax.get_ylabel() for ax in axes] == ["Rate", "", ""]
    assert [twin.get_ylabel() for twin in twins] == ["", "", "Count"]
    assert all(visible_tick_labels(titles_only.fig, ax.yaxis) for ax in [*axes, *twins])
    ticks_only = draw_grouped_means(data, outer_labels=False, outer_ticks=True, **shared)
    axes, twins = ticks_only.axes[0], ticks_only.count_axes[0]
    assert [ax.get_ylabel() for ax in axes] == ["Rate"] * 3
    assert [bool(visible_tick_labels(ticks_only.fig, ax.yaxis)) for ax in axes] == [True, False, False]
    assert [bool(visible_tick_labels(ticks_only.fig, twin.yaxis)) for twin in twins] == [False, False, True]


def test_fixed_primary_range_applies_to_every_panel():
    data = summarize_grouped_means(facet_frame(), x="x", y=["y", "p"], col="f", bins={"x": BinSpec.step(.25)})
    result = draw_grouped_means(data, wrap=3, compact_cols=False, ylim=(-.01, .2))
    assert [ax.get_ylim() for ax in result.axes.flat[:len(LEVELS)]] == [(-.01, .2)] * len(LEVELS)
    for bad in ((.2, .1), (0, float("inf")), (0,)):
        with pytest.raises(ValueError, match="ylim"):
            draw_grouped_means(data, wrap=3, ylim=bad)


def test_facet_panels_align_primary_scale_across_pages():
    diagnostics = MortgageDiagnostics(facet_frame(), MortgageColnames(
        response="y", model_preds={"Model": "p"}, incentive=("x", .25)))
    panels = diagnostics.facet_panels("incentive", "f", n_cols=2, align_ylim=True)
    first = panels.draw(["a", "b", "c", "d"], panel_size=(3, 2))
    rest = panels.draw(["e"], panel_size=(3, 2))
    primaries = [ax for ax in first.axes[:4] + rest.axes[:1]]
    limits = {ax.get_ylim() for ax in primaries}
    assert len(limits) == 1
    low, high = limits.pop()
    means = np.concatenate([line.get_ydata() for ax in primaries for line in ax.get_lines()]).astype(float)
    means = means[np.isfinite(means)]
    pad = (means.max() - means.min()) * .05
    assert (low, high) == pytest.approx((means.min() - pad, means.max() + pad))
    assert visible_tick_labels(first, first.axes[1].yaxis) == []
    assert visible_tick_labels(first, first.axes[0].yaxis)
    every = diagnostics.facet_panels("incentive", "f", n_cols=2, align_ylim=True, outer_ticks=False)
    shown = every.draw(["a", "b"], panel_size=(3, 2))
    assert {ax.get_ylim() for ax in shown.axes[:2]} == {(low, high)}
    assert all(visible_tick_labels(shown, ax.yaxis) for ax in shown.axes)  # both panels, both axes
    assert shown.axes[1].get_ylabel() == "" and shown.axes[0].get_ylabel()
    default = diagnostics.facet_panels("incentive", "f", n_cols=2)
    assert len({ax.get_ylim() for ax in default.draw(["a", "b"], panel_size=(3, 2)).axes[:2]}) == 2


def test_facet_panels_keep_one_count_scale_across_pages():
    diagnostics = MortgageDiagnostics(facet_frame(), MortgageColnames(
        response="y", model_preds={"Model": "p"}, incentive=("x", .25)), theme=PRINT_THEME)
    panels = diagnostics.facet_panels("incentive", "f", facet_label="Group", n_cols=2)
    assert panels.panels == tuple(LEVELS) and panels.n_cols == 2
    first = panels.draw(["a", "b", "c", "d"], panel_size=(3, 2))
    rest = panels.draw(["e"], panel_size=(3, 2))
    assert tuple(first.get_size_inches()) == (6, 4) and tuple(rest.get_size_inches()) == (6, 2)
    # Each grid's slots come first, then one count axis per drawn panel.
    count_axes = first.axes[4:] + rest.axes[2:]
    assert len(count_axes) == 5 and len({ax.get_ylim() for ax in count_axes}) == 1
    assert first.axes[0].get_title() == "Group: a"
    assert isinstance(first.axes[0].yaxis.get_major_formatter(), StepPercentFormatter)
    assert first.axes[0].yaxis.get_ticklabels()[0].get_fontsize() == PRINT_THEME.tick_labelsize
    with pytest.raises(TypeError, match="figsize"):
        diagnostics.facet_panels("incentive", "f", figsize=(1, 1))
    fig, _ = diagnostics.plot("incentive", title=None)
    assert fig.get_suptitle() == ""


def test_implied_actual_panels_draw_a_subset_with_one_figure_legend():
    from quantbullet.linear_product_model import LinearProductModelToolkit
    from quantbullet.model.feature import DataType, Feature, FeatureRole, FeatureSpec
    toolkit = LinearProductModelToolkit(FeatureSpec([
        Feature("x", DataType.FLOAT, FeatureRole.MODEL_INPUT), Feature("z", DataType.FLOAT, FeatureRole.MODEL_INPUT),
        Feature("y", DataType.FLOAT, FeatureRole.TARGET)]))
    frame = pd.DataFrame({"feature_value": [1, 1, 2], "y": [0., 1., 0.], "m": [.2, .4, .3],
                          "model_pred": [1., 2., 1.5], "w": [1., 3., 1.]})
    with patch.object(toolkit, "compute_implied_actual_data", return_value={"x": frame, "z": frame.copy()}):
        panels = toolkit.implied_actual_panels(SimpleNamespace(loss_="mse"), None,
                                               bin_config={"x": "discrete", "z": "discrete"}, n_cols=2)
    assert panels.panels == ("x", "z")
    fig = panels.draw(["z"], panel_size=(3, 2))
    assert tuple(fig.get_size_inches()) == (6, 2)
    assert len(fig.legends) == 1 and fig.axes[0].get_legend() is None
    assert fig.axes[0].get_xlabel() == "z"
