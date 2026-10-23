"""Two-way diagnostic figures: heatmap cells, slice panels, overlays and theme."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from quantbullet.diagnostics import TwoWayDiagnostic  # noqa: E402
from quantbullet.plot import PRINT_THEME, BinSpec  # noqa: E402
from tests.diagnostics.helpers import X_KNOTS, Z_KNOTS, tilt_nodes, two_way_frame  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def frame():
    return two_way_frame(n=40_000, seed=12, nodes=tilt_nodes())


def diagnostic(frame):
    return TwoWayDiagnostic(frame, x="balance", z="incentive", y="y", pred="pred", weight="w", x_scale="log",
                            labels={"balance": "current balance"})


def texts(ax):
    return [text.get_text() for text in ax.texts]


def has_line_at_one(ax):
    return any(np.allclose(line.get_ydata(), 1) and len(line.get_ydata()) == 2 for line in ax.get_lines())


def test_ae_figure_has_a_heatmap_and_one_slice_per_bin(frame):
    ae = diagnostic(frame).ae(x_bins=BinSpec.edges([3e4, 1e5, 3e5, 1e6]),
                              z_bins=BinSpec.edges([-6., -5.4, -4., -2., 0.]))
    plot = ae.plot(min_rows=500)
    assert len(plot.slice_axes) == 3 and plot.result is ae
    rows = ae.pivot("rows").to_numpy()
    assert texts(plot.heatmap_ax).count("–") == int((rows < 500).sum()) > 0
    assert all(has_line_at_one(ax) for ax in plot.slice_axes)
    assert plot.slice_axes[0].get_title(loc="left").startswith("current balance: [30K, 100K]")
    assert len(plot.fig.legends) == 1
    by_z = ae.plot(slices="z")
    assert len(by_z.slice_axes) == 4
    # Balance runs along these slices on a log axis, with ticks at round 1-2-5 values.
    axis = by_z.slice_axes[0]
    assert axis.get_xscale() == "log"
    low, high = axis.get_xlim()
    labels = [axis.xaxis.get_major_formatter()(tick) for tick in axis.get_xticks() if low <= tick <= high]
    assert labels == ["50K", "100K", "200K", "500K"]   # the axis ends at the largest balance, just under 1M


def test_correction_figure_skips_the_reference_row_and_draws_overlays(frame):
    diag = diagnostic(frame)
    fit = diag.correction(x_knots=X_KNOTS, z_knots=Z_KNOTS, x_reference=2.5e5, z_reference=-3)
    subset = diagnostic(frame.filter(frame["incentive"] > -4)).correction(
        x_knots=(5e4, 2.5e5, 8e5), z_knots=(-5., -3., -.5), x_reference=2.5e5, z_reference=-3)
    plot = fit.plot(overlays={"subset": subset}, label="all")
    assert len(plot.slice_axes) == len(X_KNOTS) - 1
    assert texts(plot.heatmap_ax).count("1 · ref") == int((~fit.free).sum())
    assert all(any(len(line.get_xdata()) == 200 for line in ax.get_lines()) for ax in plot.slice_axes)
    assert [text.get_text() for text in plot.fig.legends[0].get_texts()] == ["all (±95%)", "subset"]
    assert len(fit.plot(slices="z").slice_axes) == len(Z_KNOTS) - 1
    free = diag.correction(x_knots=X_KNOTS, z_knots=Z_KNOTS)
    assert len(free.plot().slice_axes) == len(X_KNOTS)


def test_overlays_must_share_columns_and_the_theme_sets_fonts(frame):
    fit = diagnostic(frame).correction(x_knots=X_KNOTS, z_knots=Z_KNOTS, x_reference=2.5e5, z_reference=-3)
    other = TwoWayDiagnostic(frame, x="incentive", z="balance", y="y", pred="pred").correction(
        x_knots=Z_KNOTS, z_knots=(5e4, 8e5))
    with pytest.raises(ValueError, match="same x and z"):
        fit.plot(overlays={"other": other})
    plot = fit.plot(theme=PRINT_THEME)
    assert plot.heatmap_ax.get_yticklabels()[0].get_fontsize() == PRINT_THEME.tick_labelsize
    with pytest.raises(ValueError, match="slices"):
        fit.plot(slices="y")


def test_fixed_color_and_y_ranges_let_two_figures_compare(frame):
    diag = diagnostic(frame)
    ae = diag.ae(x_bins=BinSpec.edges([3e4, 1e5, 3e5, 1e6]), z_bins=BinSpec.edges([-6., -4., -2., 0.]))
    fit = diag.correction(x_knots=X_KNOTS, z_knots=Z_KNOTS, x_reference=2.5e5, z_reference=-3)
    for plot in (ae.plot(color_range=1.5, y_range=(.6, 1.6)), fit.plot(color_range=1.5, y_range=(.6, 1.6))):
        image = plot.heatmap_ax.images[0]
        assert image.get_clim() == pytest.approx((-np.log(1.5), np.log(1.5)))
        colorbar_ticks = image.colorbar.ax.get_yticks()
        assert np.all(np.abs(colorbar_ticks) <= np.log(1.5) + 1e-12)
        assert all(ax.get_ylim() == pytest.approx((.6, 1.6)) for ax in plot.slice_axes)
    # Without them, both follow the data.
    assert fit.plot().heatmap_ax.images[0].get_clim()[1] == pytest.approx(np.abs(np.log(fit.nodes)).max())
    for bad in (dict(color_range=1.), dict(color_range=np.inf), dict(y_range=(1.5, .5)), dict(y_range=(0, np.nan))):
        with pytest.raises(ValueError, match="color_range|y_range"):
            ae.plot(**bad)
