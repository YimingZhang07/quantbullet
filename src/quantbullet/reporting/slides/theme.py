"""Colors and fonts of a slide deck: the one place they are defined.

The template's SCSS, matplotlib's rcParams and the Plotly template are all
generated from a :class:`SlideTheme`, so charts, tables and slides share one
palette.
"""

from __future__ import annotations

import dataclasses
import importlib.resources
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

_HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_COLOR_FIELDS = ("primary", "accent", "secondary", "neutral", "ink", "muted", "bg_alt",
                 "positive", "negative")
# Families bundled in quantbullet/reporting/fonts: family name -> file stem prefix.
BUNDLED_FONTS = {"Source Sans 3": "SourceSans3", "Source Serif 4": "SourceSerif4",
                 "Roboto": "Roboto", "Source Code Pro": "SourceCodePro"}


@dataclass(frozen=True)
class SlideTheme:
    """Design tokens for slides, charts and tables.

    Colors are hex strings. ``palette`` is the categorical order for chart
    series; ``None`` uses (primary, accent, secondary, neutral, positive,
    negative). ``font_family`` is a CSS family name; a family bundled with
    quantbullet (``BUNDLED_FONTS``) is embedded in the deck so it renders
    offline, any other is left to the viewer's system unless ``webfont_url``
    loads it (a stylesheet URL such as a Google Fonts link).
    ``heading_font_family`` is the family of slide headings and large numbers
    (``None``: the body family), embedded the same way. ``base_font_px`` is the
    body text size on the 1600 x 900 slide. ``y_axis_side`` puts the value axis
    of charts on the ``"left"`` or the ``"right"`` (an editorial habit).
    """

    primary: str = "#0b3d6e"
    accent: str = "#f2a900"
    secondary: str = "#5b8db8"
    neutral: str = "#9aa5b1"
    ink: str = "#1f2937"
    muted: str = "#6b7280"
    bg_alt: str = "#e8eef5"
    positive: str = "#2e7d6b"
    negative: str = "#b5503c"
    palette: tuple[str, ...] | None = None
    font_family: str = "Source Sans 3"
    heading_font_family: str | None = None
    webfont_url: str | None = None
    base_font_px: int = 32
    y_axis_side: Literal["left", "right"] = "left"

    def __post_init__(self):
        for name in _COLOR_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not _HEX.match(value):
                raise ValueError(f"{name} must be a hex color such as '#0b3d6e'; got {value!r}.")
        for value in self.palette or ():
            if not _HEX.match(value):
                raise ValueError(f"palette colors must be hex strings; got {value!r}.")
        if self.palette is not None:
            object.__setattr__(self, "palette", tuple(self.palette))
        if self.y_axis_side not in ("left", "right"):
            raise ValueError(f"y_axis_side must be 'left' or 'right'; got {self.y_axis_side!r}.")

    def replace(self, **changes) -> SlideTheme:
        """Return a copy with ``changes`` applied."""
        return dataclasses.replace(self, **changes)

    @property
    def colors(self) -> tuple[str, ...]:
        """The categorical palette for chart series."""
        if self.palette:
            return self.palette
        return (self.primary, self.accent, self.secondary, self.neutral, self.positive,
                self.negative)

    @property
    def heading_family(self) -> str:
        """The family of headings: ``heading_font_family``, or the body family when unset."""
        return self.heading_font_family or self.font_family

    # ---- CSS side --------------------------------------------------------------------------

    def to_scss(self) -> str:
        """SCSS variable definitions (``$qb-primary`` ...) for the template's defaults layer."""
        lines = [f"$qb-{name.replace('_', '-')}: {getattr(self, name)};" for name in _COLOR_FIELDS]
        fallback = '"Helvetica Neue", Arial, sans-serif'
        lines.append(f'$qb-font-family: "{self.font_family}", {fallback};')
        lines.append(f'$qb-heading-font-family: "{self.heading_family}", {fallback};')
        lines.append(f"$qb-font-size-root: {int(self.base_font_px)}px;")
        return "\n".join(lines) + "\n"

    def font_files(self, family: str | None = None) -> dict[str, Path]:
        """Bundled font files of ``family`` (default ``font_family``) keyed by weight/style.

        Keys are ``"Regular"``, ``"Bold"`` and ``"Italic"``; {} when the family is not bundled.
        """
        stem = BUNDLED_FONTS.get(family or self.font_family)
        if stem is None:
            return {}
        folder = importlib.resources.files("quantbullet.reporting") / "fonts"
        files = {}
        for key in ("Regular", "Bold", "Italic"):
            path = Path(str(folder / f"{stem}-{key}.ttf"))
            if path.is_file():
                files[key] = path
        return files

    def embedded_fonts(self) -> list[tuple[str, str, Path]]:
        """``(family, style key, file)`` for every bundled font file a deck embeds.

        The body family brings Regular, Bold and Italic; a separate heading family brings
        Regular and Bold.
        """
        fonts = [(self.font_family, key, path) for key, path in self.font_files().items()]
        if self.heading_family != self.font_family:
            fonts += [(self.heading_family, key, path)
                      for key, path in self.font_files(self.heading_family).items()
                      if key in ("Regular", "Bold")]
        return fonts

    # ---- matplotlib ------------------------------------------------------------------------

    def mpl_rc(self) -> dict:
        """rcParams for charts shown on a 1600 x 900 slide."""
        from cycler import cycler

        rc = {
            "axes.prop_cycle": cycler(color=list(self.colors)),
            "axes.edgecolor": self.neutral,
            "axes.labelcolor": self.ink,
            "axes.titlecolor": self.ink,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 14,
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
            "xtick.color": self.muted,
            "ytick.color": self.muted,
            "xtick.labelcolor": self.ink,
            "ytick.labelcolor": self.ink,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.frameon": False,
            "legend.fontsize": 12,
            "lines.linewidth": 2.2,
            "figure.figsize": (12, 5.5),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.constrained_layout.use": True,
            "font.family": [self.font_family, "DejaVu Sans"],
            "svg.fonttype": "none",
        }
        if self.y_axis_side == "right":
            rc.update({"axes.spines.left": False, "ytick.left": False, "ytick.right": False,
                       "ytick.labelleft": False, "ytick.labelright": True})
        return rc

    @contextmanager
    def matplotlib(self) -> Iterator[None]:
        """Apply :meth:`mpl_rc` inside a ``with`` block; bundled fonts are registered first."""
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        registered = {Path(font.fname).resolve() for font in font_manager.fontManager.ttflist}
        for _, _, path in self.embedded_fonts():
            if path.resolve() not in registered:
                font_manager.fontManager.addfont(str(path))
        with plt.rc_context(self.mpl_rc()):
            yield

    # ---- Plotly ----------------------------------------------------------------------------

    def plotly_template(self):
        """A Plotly layout template in this theme; layer it on ``plotly_white``."""
        import plotly.graph_objects as go

        axis = dict(gridcolor="#e5e7eb", linecolor=self.neutral, zerolinecolor="#d1d5db",
                    tickfont=dict(size=14, color=self.ink), title=dict(font=dict(size=16)))
        yaxis = dict(axis, side=self.y_axis_side)
        return go.layout.Template(layout=dict(
            colorway=list(self.colors),
            font=dict(family=f"{self.font_family}, Helvetica Neue, Arial, sans-serif", size=16,
                      color=self.ink),
            title=dict(font=dict(size=20, color=self.ink), x=0, xanchor="left"),
            margin=(dict(l=20, r=70, t=60, b=50) if self.y_axis_side == "right"
                    else dict(l=60, r=20, t=60, b=50)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        font=dict(size=14)),
            hoverlabel=dict(bgcolor="white", bordercolor=self.neutral,
                            font=dict(family=self.font_family, size=14, color=self.ink)),
            xaxis=axis,
            yaxis=yaxis,
            paper_bgcolor="white",
            plot_bgcolor="white",
        ))

    def _full_plotly_template(self):
        import plotly.graph_objects as go
        import plotly.io as pio

        template = go.layout.Template(pio.templates["plotly_white"])  # a copy
        template.layout.update(self.plotly_template().layout)
        return template

    @contextmanager
    def plotly(self) -> Iterator[None]:
        """Make this theme Plotly's default template inside a ``with`` block.

        Plotly Express picks series colors when a figure is created, so create ``px`` figures
        inside this block to give them the theme palette from the start.
        """
        import plotly.io as pio

        previous = pio.templates.default
        pio.templates.default = self._full_plotly_template()
        try:
            yield
        finally:
            pio.templates.default = previous

    def style_plotly(self, fig):
        """Apply ``plotly_white`` plus :meth:`plotly_template` to ``fig`` in place and return it.

        Series colors that Plotly Express took from Plotly's default palette are swapped for
        the theme palette, position by position; colors set any other way are kept.
        """
        import plotly.io as pio

        fig.update_layout(template=self._full_plotly_template())
        default = [c.lower() for c in pio.templates["plotly"].layout.colorway]
        swap = {c: self.colors[i % len(self.colors)] for i, c in enumerate(default)}
        for trace in fig.data:
            for part in ("line", "marker"):
                obj = getattr(trace, part, None)
                color = getattr(obj, "color", None) if obj is not None else None
                if isinstance(color, str) and color.lower() in swap:
                    obj.color = swap[color.lower()]
        return fig


DEFAULT_THEME = SlideTheme()
