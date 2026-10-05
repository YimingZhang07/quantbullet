"""Frames and checks shared by plot tests."""
import numpy as np
import polars as pl

LEVELS = ["a", "b", "c", "d", "e"]


def facet_frame(n=600):
    rng = np.random.default_rng(0)
    return pl.DataFrame({"x": rng.uniform(0, 1, n), "y": rng.random(n) * .1, "p": rng.random(n) * .1,
                         "f": rng.choice(LEVELS, n)}).with_columns(pl.col("f").cast(pl.Enum(LEVELS)))


def visible_tick_labels(fig, axis):
    fig.canvas.draw()
    return [label.get_text() for label in axis.get_ticklabels() if label.get_visible()]
