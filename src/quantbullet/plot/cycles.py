from .colors import EconomistBrandColor
from contextlib import contextmanager
from cycler import cycler
import matplotlib.pyplot as plt

# Colors and linestyles
_ECONOMIST_COLORS = [
    EconomistBrandColor.CHICAGO_45,   # blue
    EconomistBrandColor.SINGAPORE_55, # orange
    EconomistBrandColor.HONG_KONG_45, # green
    EconomistBrandColor.TOKYO_45,     # red
    EconomistBrandColor.LONDON_20,    # purple
    EconomistBrandColor.NEW_YORK_55,  # yellow
]

_BASE_LINESTYLES = ['-', '--']

# Create cycle: all colors with first style, then all colors with second style
_ECONOMIST_CYCLE = cycler(
    color=_ECONOMIST_COLORS * len(_BASE_LINESTYLES),
    linestyle=[style for style in _BASE_LINESTYLES for _ in _ECONOMIST_COLORS]
)

@contextmanager
def use_economist_cycle():
    orig_cycle = plt.rcParams['axes.prop_cycle']
    try:
        plt.rcParams['axes.prop_cycle'] = _ECONOMIST_CYCLE
        yield
    finally:
        plt.rcParams['axes.prop_cycle'] = orig_cycle
