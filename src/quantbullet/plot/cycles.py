from .colors import EconomistBrandColor
from contextlib import contextmanager
from cycler import cycler
import matplotlib.pyplot as plt

# Colors and linestyles
ECONOMIST_COLORS = [
    EconomistBrandColor.CHICAGO_45,   # blue
    EconomistBrandColor.SINGAPORE_55, # orange
    EconomistBrandColor.HONG_KONG_45, # green
    EconomistBrandColor.TOKYO_45,     # red
    EconomistBrandColor.NEW_YORK_55,  # yellow
    EconomistBrandColor.PURPLE_70,    # purple
    EconomistBrandColor.LONDON_20,    # Black/Grey
]

# Export for external use (e.g., in binned plots)
ECONOMIST_LINE_COLORS = ECONOMIST_COLORS

_BASE_LINESTYLES = ['-', '--']

# Create cycle: all colors with first style, then all colors with second style
_ECONOMIST_CYCLE = cycler(
    color=ECONOMIST_COLORS * len(_BASE_LINESTYLES),
    linestyle=[style for style in _BASE_LINESTYLES for _ in ECONOMIST_COLORS]
)

@contextmanager
def use_economist_cycle():
    orig_cycle = plt.rcParams['axes.prop_cycle']
    try:
        plt.rcParams['axes.prop_cycle'] = _ECONOMIST_CYCLE
        yield
    finally:
        plt.rcParams['axes.prop_cycle'] = orig_cycle

ECONOMIST_SIDE_BY_SIDE_COLORS = [
    "#106ea0",
    "#32c0d2",
    "#e0b165",
    "#00969e",
    "#963c4c",
    "#ab8b95",
]

ECONOMIST_SIDE_BY_SIDE_CYCLE = cycler(
    color       = ECONOMIST_SIDE_BY_SIDE_COLORS * len( _BASE_LINESTYLES ),
    linestyle   = [ style for style in _BASE_LINESTYLES for _ in ECONOMIST_SIDE_BY_SIDE_COLORS ]
)

@contextmanager
def use_economist_side_by_side_cycle():
    orig_cycle = plt.rcParams['axes.prop_cycle']
    try:
        plt.rcParams['axes.prop_cycle'] = ECONOMIST_SIDE_BY_SIDE_CYCLE
        yield
    finally:
        plt.rcParams['axes.prop_cycle'] = orig_cycle