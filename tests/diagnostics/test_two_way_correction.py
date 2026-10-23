"""TwoWayCorrection: the bilinear grid, recovery of known surfaces, references, losses and checks."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from quantbullet.diagnostics import TwoWayDiagnostic
from quantbullet.diagnostics.two_way.correction import _Axis, _design
from tests.diagnostics.helpers import X_KNOTS, Z_KNOTS, tensor_basis, tilt_nodes, two_way_frame


def diagnostic(frame, **kwargs):
    return TwoWayDiagnostic(frame, x="balance", z="incentive", y="y", pred="pred", weight="w", x_scale="log",
                            **kwargs)


def test_the_design_is_the_tensor_product_of_tent_bases():
    rng = np.random.default_rng(0)
    x = np.exp(rng.uniform(np.log(2e4), np.log(2e6), 2_000))   # beyond both end knots
    z = rng.uniform(-7, 1, 2_000)
    x_basis, _ = _Axis(X_KNOTS, "log", "x_knots").basis(x)
    z_basis, clamped = _Axis(Z_KNOTS, "linear", "z_knots").basis(z)
    design = _design(x_basis, z_basis)
    assert design.shape == (2_000, len(X_KNOTS) * len(Z_KNOTS)) and (np.diff(design.indptr) == 4).all()
    np.testing.assert_allclose(design.toarray(), tensor_basis(x, z), atol=1e-12)
    assert clamped == int(((z < -5) | (z > -.5)).sum())


def test_anchored_poisson_recovers_an_interaction():
    truth = tilt_nodes(theta=.08)
    frame = two_way_frame(n=400_000, seed=4, nodes=truth)
    fit = diagnostic(frame).correction(x_knots=X_KNOTS, z_knots=Z_KNOTS, x_reference=2.5e5, z_reference=-3,
                                       ridge=0)
    assert fit.converged and fit.free.sum() == 20
    assert np.all(np.abs(fit.nodes - truth)[fit.free] < 4 * fit.se[fit.free])
    assert (fit.nodes[2] == 1).all() and (fit.nodes[:, 2] == 1).all() and (fit.se[~fit.free] == 0).all()
    on_lines = fit.surface([2.5e5, 2.5e5, 7e4, 6e5], [-4.4, -.7, -3, -3])
    np.testing.assert_allclose(on_lines, 1, atol=1e-12)
    assert fit.deviance["fitted"] < fit.deviance["baseline"] and fit.deviance["change"] < 0
    assert fit.clamped == {"x": int(((frame["balance"] < 5e4) | (frame["balance"] > 8e5)).sum()),
                           "z": int(((frame["incentive"] < -5) | (frame["incentive"] > -.5)).sum())}


def test_without_references_the_surface_carries_the_level():
    frame = two_way_frame(n=300_000, seed=5, pred_scale=.8)
    fit = diagnostic(frame).correction(x_knots=X_KNOTS, z_knots=Z_KNOTS)
    assert fit.free.all() and fit.x_reference is None and fit.z_reference is None
    assert np.all(np.abs(fit.nodes - 1.25) < 4 * fit.se)


def test_one_reference_fixes_only_its_line():
    frame = two_way_frame(n=100_000, seed=6)
    fit = diagnostic(frame).correction(x_knots=X_KNOTS, z_knots=Z_KNOTS, z_reference=-3)
    assert (~fit.free).sum() == len(X_KNOTS) and (fit.nodes[:, 2] == 1).all()
    assert np.all(np.abs(fit.nodes - 1)[fit.free] < 4 * fit.se[fit.free])


def test_mse_equals_the_penalized_least_squares_solution():
    truth = tilt_nodes(theta=.1)
    frame = two_way_frame(n=50_000, seed=7, nodes=truth, loss="mse")
    ridge = .05
    fit = diagnostic(frame).correction(x_knots=X_KNOTS, z_knots=Z_KNOTS, x_reference=2.5e5, z_reference=-3,
                                       loss="mse", ridge=ridge)
    x, z, y, m, w = (frame[name].to_numpy() for name in ("balance", "incentive", "y", "pred", "w"))
    design = m[:, None] * tensor_basis(x, z)
    free = fit.free.ravel()
    gram = design[:, free].T @ (w[:, None] * design[:, free])
    scale = np.diag(gram).mean()
    target = y - design[:, ~free].sum(axis=1)
    solved = np.linalg.solve(gram + ridge * scale * np.eye(free.sum()),
                             design[:, free].T @ (w * target) + ridge * scale)
    assert fit.penalty_scale == pytest.approx(scale, rel=1e-10)
    np.testing.assert_allclose(fit.nodes.ravel()[free], solved, rtol=1e-7)   # L-BFGS-B, not a direct solve


def test_a_large_ridge_pulls_every_node_to_one():
    frame = two_way_frame(n=50_000, seed=8, nodes=tilt_nodes(theta=.2))
    fit = diagnostic(frame).correction(x_knots=X_KNOTS, z_knots=Z_KNOTS, x_reference=2.5e5, z_reference=-3,
                                       ridge=1e6)
    np.testing.assert_allclose(fit.nodes, 1, atol=1e-4)


def test_poisson_nodes_stay_positive_where_there_are_no_events():
    frame = two_way_frame(n=50_000, seed=9).with_columns(
        pl.when(pl.col("incentive") > -1.5).then(0.).otherwise(pl.col("y")).alias("y"))
    # Without a ridge the likelihood keeps falling toward D = 0 there: nodes reach the bound but stay positive,
    # and the nodes still moving are not held back by them.
    fit = diagnostic(frame).correction(x_knots=X_KNOTS, z_knots=Z_KNOTS, x_reference=2.5e5, z_reference=-3,
                                       ridge=0)
    assert fit.converged and (fit.nodes > 0).all() and (fit.nodes[:, -1][fit.free[:, -1]] < 1e-3).all()
    # A strong ridge pulls them back toward 1: they settle where the pull balances the expected events.
    ridged = diagnostic(frame).correction(x_knots=X_KNOTS, z_knots=Z_KNOTS, x_reference=2.5e5, z_reference=-3,
                                          ridge=100)
    assert ridged.converged and (ridged.nodes[:, -1][ridged.free[:, -1]] > .5).all()


def test_tables_prediction_and_deviance():
    frame = two_way_frame(n=60_000, seed=10, nodes=tilt_nodes())
    diag = diagnostic(frame, labels={"balance": "current balance"})
    fit = diag.correction(x_knots=X_KNOTS, z_knots=Z_KNOTS, x_reference=2.5e5, z_reference=-3)
    lower, upper, nodes, se = (fit.table(value, ci=.9) for value in ("lower", "upper", "nodes", "se"))
    np.testing.assert_allclose((upper - lower).to_numpy(), 2 * 1.6448536 * se.to_numpy(), rtol=1e-6)
    z_score = fit.table("z_score")
    assert z_score.isna().to_numpy().sum() == 10
    np.testing.assert_allclose(z_score.to_numpy()[fit.free], (nodes.to_numpy() - 1)[fit.free] / se.to_numpy()[fit.free])
    assert nodes.index.name == "current balance" and list(nodes.columns) == list(Z_KNOTS)
    assert fit.table("rows").to_numpy().sum() == pytest.approx(60_000)
    with pytest.raises(ValueError, match="value"):
        fit.table("lift")

    x, z, y, m, w = (frame[name].to_numpy() for name in ("balance", "incentive", "y", "pred", "w"))
    as_pandas = pd.DataFrame({"balance": x, "incentive": z})
    D = fit.predict(as_pandas)
    np.testing.assert_allclose(D, tensor_basis(x, z) @ fit.nodes.ravel(), rtol=1e-12)
    np.testing.assert_allclose(fit.predict(pl.from_dict({"balance": x, "incentive": z})), D)
    for mu, key in ((m, "baseline"), (m * D, "fitted")):
        deviance = 2 * np.sum(w * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))) / w.sum()
        assert fit.deviance[key] == pytest.approx(deviance, rel=1e-10)
    assert np.isnan(fit.surface([-1., np.nan], [-3., -3.])).all()


def test_zero_predictions_and_bad_arguments():
    frame = two_way_frame(n=20_000, seed=11)
    zeroed = frame.with_columns(pl.when(pl.int_range(pl.len()) < 5).then(0.).otherwise(pl.col("pred")).alias("pred"),
                                pl.when(pl.int_range(pl.len()) < 5).then(0.).otherwise(pl.col("y")).alias("y"))
    fit = diagnostic(zeroed).correction(x_knots=X_KNOTS, z_knots=Z_KNOTS)
    assert fit.excluded == {"zero_pred": 5} and fit.n_obs == 20_000 - 5
    with pytest.raises(ValueError, match="pred = 0 and y > 0"):
        diagnostic(zeroed.with_columns(y=1.)).correction(x_knots=X_KNOTS, z_knots=Z_KNOTS)
    with pytest.raises(ValueError, match="y >= 0"):
        diagnostic(frame.with_columns(y=-pl.col("y") - 1)).correction(x_knots=X_KNOTS, z_knots=Z_KNOTS)
    diag = diagnostic(frame)
    for kwargs, match in (
            (dict(x_reference=2e5), "one of the knots"),
            (dict(x_knots=(5e4, 4e4, 8e5)), "strictly increasing"),
            (dict(x_knots=(0., 1e5)), "positive"),
            (dict(loss="gamma"), "loss"),
            (dict(ridge=-1.), "ridge"),
            (dict(z_knots=(-5., -3., 5., 9.), ridge=0), "without data"),
    ):
        with pytest.raises(ValueError, match=match):
            diag.correction(**{"x_knots": X_KNOTS, "z_knots": Z_KNOTS, **kwargs})
