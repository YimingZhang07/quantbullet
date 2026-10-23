"""TwoWayAE: binned actual / expected against a direct pandas computation."""
import numpy as np
import pandas as pd
import pytest

from quantbullet.diagnostics import TwoWayDiagnostic
from quantbullet.plot import BinSpec
from tests.diagnostics.helpers import two_way_frame

X_EDGES = [3e4, 1.5e5, 3.5e5, 1e6]
Z_EDGES = [-6., -4., -2., 0.]


def diagnostic(frame, weight="w"):
    return TwoWayDiagnostic(frame, x="balance", z="incentive", y="y", pred="pred", weight=weight, x_scale="log",
                            labels={"balance": "current balance"})


def test_cells_match_a_direct_groupby():
    frame = two_way_frame(n=20_000, seed=1)
    ae = diagnostic(frame).ae(x_bins=BinSpec.edges(X_EDGES), z_bins=BinSpec.edges(Z_EDGES))
    data = pd.DataFrame({name: frame[name].to_numpy() for name in frame.columns})
    data["x_bin"] = pd.cut(data["balance"], X_EDGES, include_lowest=True)
    data["z_bin"] = pd.cut(data["incentive"], Z_EDGES, include_lowest=True)
    data = data.assign(wy=data.w * data.y, wp=data.w * data.pred, sq=(data.w * (data.y - data.pred)) ** 2)
    expected = data.groupby(["x_bin", "z_bin"], observed=True).agg(
        rows=("y", "size"), weight_sum=("w", "sum"), actual=("wy", "sum"), expected=("wp", "sum"), sq=("sq", "sum"))
    table = ae.table
    assert len(table) == 9 and ae.n_obs == 20_000 and ae.excluded == {}
    np.testing.assert_array_equal(table["rows"], expected["rows"])
    for column in ("weight_sum", "actual", "expected"):
        np.testing.assert_allclose(table[column], expected[column], rtol=1e-12)
    np.testing.assert_allclose(table["ae"], expected["actual"] / expected["expected"], rtol=1e-12)
    np.testing.assert_allclose(table["ae_se"], np.sqrt(expected["sq"]) / expected["expected"], rtol=1e-12)
    np.testing.assert_allclose(table["actual_mean"], expected["actual"] / expected["weight_sum"], rtol=1e-12)
    np.testing.assert_allclose(table["pred_mean"], expected["expected"] / expected["weight_sum"], rtol=1e-12)
    assert [str(level) for level in table["x_bin"].unique()] == [str(level) for level in ae.x_levels]


def test_empty_cells_and_rows_outside_the_edges():
    frame = two_way_frame(n=5_000, seed=2)   # incentives start at -5.5, so [-6, -5.6] is always empty
    ae = diagnostic(frame).ae(x_bins=BinSpec.edges([3e4, 1.5e5, 3.5e5]), z_bins=BinSpec.edges([-6., -5.6, -3., 0.]))
    outside = int((frame["balance"] > 3.5e5).sum())
    assert ae.excluded == {"outside_bins": outside} and ae.n_obs == 5_000 - outside
    assert len(ae.table) == 6  # every z bin of every observed x bin
    empty = ae.table.loc[ae.table["rows"] == 0]
    assert len(empty) == 2 and (empty["z_bin"] == ae.z_levels[0]).all()
    assert empty["ae"].isna().all() and (empty["expected"] == 0).all()


def test_pivot_and_unit_weights():
    frame = two_way_frame(n=5_000, seed=3)
    weighted = diagnostic(frame.with_columns(w=1.)).ae(x_bins=BinSpec.edges(X_EDGES), z_bins=BinSpec.edges(Z_EDGES))
    plain = diagnostic(frame, weight=None).ae(x_bins=BinSpec.edges(X_EDGES), z_bins=BinSpec.edges(Z_EDGES))
    pd.testing.assert_frame_equal(weighted.pivot("ae"), plain.pivot("ae"))
    wide = plain.pivot("rows")
    assert wide.shape == (3, 3) and wide.index.name == "current balance" and wide.columns.name == "incentive"
    assert wide.to_numpy().sum() == 5_000
    with pytest.raises(ValueError, match="value"):
        plain.pivot("lift")
    with pytest.raises(TypeError, match="BinSpec"):
        diagnostic(frame).ae(x_bins=X_EDGES, z_bins=BinSpec.edges(Z_EDGES))
