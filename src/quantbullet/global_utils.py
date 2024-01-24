from contextlib import contextmanager
import matplotlib.pyplot as plt


@contextmanager
def set_figsize(width, height):
    """Temporarily set the figure size using a context manager."""
    orig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = (width, height)
    yield
    plt.rcParams["figure.figsize"] = orig_size
