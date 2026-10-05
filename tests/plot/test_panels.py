"""Panel sets, report producers that page through them, and print formatting."""
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
from quantbullet.plot.panels import PanelSet, panel_grid
from quantbullet.plot.theme import PRINT_THEME
from tests.plot.helpers import LEVELS, facet_frame, visible_tick_labels


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


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


def test_facet_panels_align_primary_scale_across_pages():
    diagnostics = MortgageDiagnostics(facet_frame(), MortgageColnames(
        response="y", model_preds={"Model": "p"}, incentive=("x", .25)))
    panels = diagnostics.facet_panels("incentive", "f", n_cols=2, y_scale="shared")
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
    every = diagnostics.facet_panels("incentive", "f", n_cols=2, y_scale="shared", y_ticks="all", bar_ticks="all")
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
    bar_axes = first.axes[4:] + rest.axes[2:]
    assert len(bar_axes) == 5 and len({ax.get_ylim() for ax in bar_axes}) == 1
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


def test_facet_panels_share_a_weight_bar_scale_across_pages():
    frame = facet_frame().with_columns(w=pl.lit(2.))
    diagnostics = MortgageDiagnostics(frame, MortgageColnames(
        response="y", model_preds={"Model": "p"}, incentive=("x", .25), weight="w"))
    panels = diagnostics.facet_panels("incentive", "f", n_cols=2, bar_value="weight")
    first, rest = panels.draw(["a", "b", "c", "d"], panel_size=(3, 2)), panels.draw(["e"], panel_size=(3, 2))
    bar_axes = first.axes[4:] + rest.axes[2:]
    tallest = max(bar.get_height() for ax in bar_axes for bar in ax.patches)
    limits = {ax.get_ylim() for ax in bar_axes}
    assert len(limits) == 1
    low, high = limits.pop()
    assert low == 0 and high == pytest.approx(tallest * 1.05)
    assert all(bar.get_height() % 2 == 0 for ax in bar_axes for bar in ax.patches)  # weight 2 per row
