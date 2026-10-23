"""Synthetic two-way data whose true multiplier surface is built from TentTransformer, not the tool under test."""
import numpy as np
import pandas as pd
import polars as pl

from quantbullet.preprocessing.transformers import TentTransformer

X_KNOTS = (5e4, 1.5e5, 2.5e5, 4.5e5, 8e5)
Z_KNOTS = (-5., -4., -3., -2., -1., -.5)


def tent(values, knots) -> np.ndarray:
    return TentTransformer(knots).fit_transform(pd.DataFrame({"v": np.asarray(values, dtype=float)}))


def tensor_basis(x, z, x_knots=X_KNOTS, z_knots=Z_KNOTS, x_log=True) -> np.ndarray:
    """Row-wise outer product of the two tent bases, x-major: the bilinear interpolation weights."""
    tx = tent(np.log(x), np.log(x_knots)) if x_log else tent(x, x_knots)
    tz = tent(z, z_knots)
    return (tx[:, :, None] * tz[:, None, :]).reshape(len(x), -1)


def tilt_nodes(theta=.06, x_knots=X_KNOTS, z_knots=Z_KNOTS, x_ref=2.5e5, z_ref=-3.) -> np.ndarray:
    """A node table that is 1 on the x_ref row and the z_ref column."""
    return np.exp(theta * np.log(np.asarray(x_knots) / x_ref)[:, None] * (np.asarray(z_knots) - z_ref)[None, :])


def two_way_frame(n=200_000, seed=0, nodes=None, rate=(.05, .15), pred_scale=1., loss="poisson"):
    """balance (log-uniform 30K-1M), incentive (uniform -5.5 to -0.25), y, pred and weight w."""
    rng = np.random.default_rng(seed)
    x = np.exp(rng.uniform(np.log(3e4), np.log(1e6), n))
    z = rng.uniform(-5.5, -.25, n)
    m = rng.uniform(*rate, n)
    D = np.ones(n) if nodes is None else tensor_basis(x, z) @ np.ravel(nodes)
    y = rng.poisson(m * D).astype(float) if loss == "poisson" else m * D + rng.normal(0, .02, n)
    return pl.DataFrame({"balance": x, "incentive": z, "y": y, "pred": m * pred_scale, "w": rng.uniform(.5, 1.5, n)})
