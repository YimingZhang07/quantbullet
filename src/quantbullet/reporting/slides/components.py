"""Slide content blocks. Each turns into Quarto Markdown with :meth:`Component.to_markdown`."""

from __future__ import annotations

import io
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import pandas as pd

from ._html import escape, html_table
from .theme import DEFAULT_THEME, SlideTheme

PLOTLY_JS = "plotly.min.js"
DEFAULT_PLOTLY_CONFIG = {"displaylogo": False, "responsive": True,
                         "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


@dataclass
class RenderContext:
    """State shared by the components of one rendering of a deck.

    ``assets_dir`` is where figure and data files go (``None`` writes nothing, for a dry
    run such as ``SlideDeck.to_qmd()``); Markdown refers to them as ``assets/<name>``.
    """

    assets_dir: Path | None = None
    theme: SlideTheme = DEFAULT_THEME
    section_titles: list[str] = field(default_factory=list)
    _counter: int = 0
    _plotly_loaded: bool = False

    def write_asset(self, stem: str, suffix: str, write: Callable[[Path], None]) -> str:
        """Write a uniquely named asset with ``write(path)``; return its path relative to the deck."""
        self._counter += 1
        name = f"{stem}-{self._counter:03d}{suffix}"
        if self.assets_dir is not None:
            self.assets_dir.mkdir(parents=True, exist_ok=True)
            write(self.assets_dir / name)
        return f"assets/{name}"

    def next_id(self, stem: str) -> str:
        self._counter += 1
        return f"{stem}-{self._counter:03d}"

    def plotly_script(self) -> str:
        """The <script> tag loading plotly.js the first time it is asked for, then ''."""
        if self._plotly_loaded:
            return ""
        self._plotly_loaded = True
        if self.assets_dir is not None:
            from plotly.offline import get_plotlyjs

            self.assets_dir.mkdir(parents=True, exist_ok=True)
            (self.assets_dir / PLOTLY_JS).write_text(get_plotlyjs(), encoding="utf-8")
        return f'<script src="assets/{PLOTLY_JS}"></script>\n'


def raw_html(markup: str) -> str:
    """Wrap HTML in a raw block so pandoc passes it through untouched."""
    return f"```{{=html}}\n{markup}\n```"


def fenced_div(body: str, *classes: str) -> str:
    attrs = " ".join(f".{c}" for c in classes)
    return f"::: {{{attrs}}}\n{body}\n:::"


class Component(ABC):
    """A block of slide content."""

    @abstractmethod
    def to_markdown(self, ctx: RenderContext) -> str:
        """Quarto Markdown for this block."""


# ---- Text ------------------------------------------------------------------------------------

class Markdown(Component):
    """Markdown passed through as written."""

    def __init__(self, text: str):
        self.text = text

    def to_markdown(self, ctx):
        return self.text.strip()


class Text(Component):
    """A paragraph of Markdown text; ``size`` is ``"small"``, ``"large"`` or None."""

    def __init__(self, text: str, *, size: Literal["small", "large"] | None = None):
        if size not in (None, "small", "large"):
            raise ValueError(f"size must be 'small', 'large' or None; got {size!r}.")
        self.text = text
        self.size = size

    def to_markdown(self, ctx):
        text = self.text.strip()
        return fenced_div(text, f"qb-text-{self.size}") if self.size else text


class Bullets(Component):
    """A bulleted list, or numbered with ``numbered=True``; ``incremental`` reveals items one by one."""

    def __init__(self, items: Sequence[str], *, numbered: bool = False, incremental: bool = False):
        self.items = [str(item) for item in items]
        if not self.items:
            raise ValueError("Bullets needs at least one item.")
        self.numbered = numbered
        self.incremental = incremental

    def to_markdown(self, ctx):
        lines = "\n".join(f"{f'{i}.' if self.numbered else '-'} {item}"
                          for i, item in enumerate(self.items, 1))
        return fenced_div(lines, "incremental") if self.incremental else lines


class Source(Component):
    """A small "Source: ..." line pinned to the bottom left of the slide."""

    def __init__(self, text: str, *, prefix: str = "Source: "):
        self.text = text
        self.prefix = prefix

    def to_markdown(self, ctx):
        return raw_html(f'<div class="qb-source">{escape(self.prefix + self.text)}</div>')


class Note(Component):
    """A small italic footnote line."""

    def __init__(self, text: str):
        self.text = text

    def to_markdown(self, ctx):
        return raw_html(f'<p class="qb-note">{escape(self.text)}</p>')


class RawHTML(Component):
    """HTML inserted as written (not escaped)."""

    def __init__(self, markup: str):
        self.markup = markup

    def to_markdown(self, ctx):
        return raw_html(self.markup)


# ---- Charts and images -----------------------------------------------------------------------

def _image_markdown(path: str, caption: str | None, width: str | None) -> str:
    # Without a width, a lone image stretches to fill the rest of the slide (Quarto's
    # auto-stretch); a width turns that off so the width is kept.
    attrs = f'{{.nostretch width="{width}"}}' if width else ""
    return f"![{caption or ''}]({path}){attrs}"


class Figure(Component):
    """A matplotlib figure, saved when the component is created (the figure can be reused).

    ``fmt`` is ``"svg"`` (sharp at any zoom) or ``"png"``; ``width`` is a CSS width such as
    ``"80%"``; ``close=True`` closes the figure in pyplot after saving it.
    """

    def __init__(self, fig, *, fmt: Literal["svg", "png"] = "svg", width: str | None = None,
                 caption: str | None = None, dpi: int = 150, close: bool = True):
        import matplotlib.pyplot as plt

        if hasattr(fig, "figure") and not hasattr(fig, "savefig"):
            fig = fig.figure  # an Axes
        if fmt not in ("svg", "png"):
            raise ValueError(f"fmt must be 'svg' or 'png'; got {fmt!r}.")
        buffer = io.BytesIO()
        with plt.rc_context({"svg.fonttype": "path"}):
            fig.savefig(buffer, format=fmt, dpi=dpi, facecolor="white", bbox_inches="tight")
        if close:
            plt.close(fig)
        self.data = buffer.getvalue()
        self.fmt = fmt
        self.width = width
        self.caption = caption

    def to_markdown(self, ctx):
        path = ctx.write_asset("fig", f".{self.fmt}", lambda p: p.write_bytes(self.data))
        return _image_markdown(path, self.caption, self.width)


class Image(Component):
    """An image file, copied into the deck's assets."""

    def __init__(self, path: str | Path, *, width: str | None = None, caption: str | None = None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.width = width
        self.caption = caption

    def to_markdown(self, ctx):
        path = ctx.write_asset("img", self.path.suffix.lower(),
                               lambda p: shutil.copyfile(self.path, p))
        return _image_markdown(path, self.caption, self.width)


class PlotlyFigure(Component):
    """An interactive Plotly figure; its data travels inside the HTML.

    ``height`` is in slide pixels (the slide is 1600 x 900). ``apply_theme`` styles a copy of
    the figure with the deck theme; the figure's own layout settings still win. ``config``
    updates the Plotly config (default: no logo, responsive, a trimmed mode bar).
    """

    def __init__(self, fig, *, height: int = 560, apply_theme: bool = True,
                 config: dict | None = None):
        self.fig = fig
        self.height = height
        self.apply_theme = apply_theme
        self.config = {**DEFAULT_PLOTLY_CONFIG, **(config or {})}

    def to_markdown(self, ctx):
        import plotly.graph_objects as go

        fig = go.Figure(self.fig)
        if self.apply_theme:
            ctx.theme.style_plotly(fig)
        div = fig.to_html(full_html=False, include_plotlyjs=False, config=self.config,
                          div_id=ctx.next_id("plotly"), default_width="100%",
                          default_height=f"{int(self.height)}px")
        return raw_html(ctx.plotly_script() + div)


def is_plotly_figure(obj) -> bool:
    return type(obj).__module__.startswith("plotly")


# ---- Tables and KPIs -------------------------------------------------------------------------

class Table(Component):
    """A DataFrame as a house-style HTML table.

    ``schema`` is a ``ColumnSchema`` or a list of ``ColumnMeta`` / ``PdfColumnMeta``: it picks,
    orders, labels and formats the columns (decimals, comma, percent, transformer, align ...).
    More than ``max_rows`` rows raises ``ValueError``. ``bold_rows`` are row positions (-1 is the
    last, e.g. a total). ``highlight_neg`` colors negative numbers.
    """

    def __init__(self, data: pd.DataFrame, *, schema=None, max_rows: int | None = 25,
                 bold_rows: Sequence[int] = (), highlight_neg: bool = True,
                 caption: str | None = None, index: bool = False):
        # Build now so a bad table fails where it is added.
        self.html = html_table(data, schema, max_rows=max_rows, bold_rows=bold_rows,
                               highlight_neg=highlight_neg, caption=caption, index=index)

    def to_markdown(self, ctx):
        return raw_html(self.html)


class Kpi(Component):
    """A KPI card: a big value, a label and an optional change.

    ``delta`` is a number (shown with its sign) or text such as ``"+2.1 bp"``; its sign colors
    it green when the move is good and red when bad. ``good="down"`` makes falling values good.
    """

    def __init__(self, value, label: str, *, delta: float | str | None = None,
                 good: Literal["up", "down"] = "up"):
        if good not in ("up", "down"):
            raise ValueError(f"good must be 'up' or 'down'; got {good!r}.")
        self.value = value
        self.label = label
        self.delta = delta
        self.good = good

    def _delta_parts(self) -> tuple[str, str]:
        delta = self.delta
        if isinstance(delta, str):
            text = delta.strip()
            sign = -1 if text[:1] in ("-", "−") else (1 if text[:1] == "+" else 0)
        else:
            delta = float(delta)  # numpy scalars too
            sign = int(delta > 0) - int(delta < 0)
            text = f"{delta:+,g}"
        if sign == 0:
            return "flat", text
        good = (sign > 0) == (self.good == "up")
        arrow = "▲" if sign > 0 else "▼"
        return ("good" if good else "bad"), f"{arrow} {text}"

    def html(self) -> str:
        parts = [f'<div class="kpi-value">{escape(self.value)}</div>',
                 f'<div class="kpi-label">{escape(self.label)}</div>']
        if self.delta is not None:
            tone, text = self._delta_parts()
            parts.append(f'<div class="kpi-delta {tone}">{escape(text)}</div>')
        return '<div class="kpi">' + "".join(parts) + "</div>"

    def to_markdown(self, ctx):
        return raw_html(self.html())


class KpiRow(Component):
    """KPI cards side by side, equal widths; items are ``Kpi`` objects or dicts of its arguments."""

    def __init__(self, kpis: Sequence[Kpi | dict]):
        self.kpis = [k if isinstance(k, Kpi) else Kpi(**k) for k in kpis]
        if not self.kpis:
            raise ValueError("KpiRow needs at least one KPI.")

    def to_markdown(self, ctx):
        return raw_html('<div class="kpi-row">' + "".join(k.html() for k in self.kpis) + "</div>")


# ---- Containers --------------------------------------------------------------------------------

class Container:
    """Holds components; every ``add``-style method returns the container, so calls chain."""

    def __init__(self):
        self.items: list[Component] = []

    def add(self, component: Component):
        if not isinstance(component, Component):
            raise TypeError(f"Expected a Component; got {type(component).__name__}.")
        self.items.append(component)
        return self

    def text(self, text: str, *, size: Literal["small", "large"] | None = None):
        return self.add(Text(text, size=size))

    def markdown(self, text: str):
        return self.add(Markdown(text))

    def bullets(self, items: Sequence[str], *, numbered: bool = False, incremental: bool = False):
        return self.add(Bullets(items, numbered=numbered, incremental=incremental))

    def chart(self, fig, **kwargs):
        """Add a matplotlib Figure / Axes (:class:`Figure`) or a Plotly figure (:class:`PlotlyFigure`)."""
        if is_plotly_figure(fig):
            return self.add(PlotlyFigure(fig, **kwargs))
        return self.add(Figure(fig, **kwargs))

    def image(self, path: str | Path, *, width: str | None = None, caption: str | None = None):
        return self.add(Image(path, width=width, caption=caption))

    def table(self, data: pd.DataFrame, **kwargs):
        """Add a :class:`Table`; see it for the keyword arguments."""
        return self.add(Table(data, **kwargs))

    def kpis(self, kpis: Sequence[Kpi | dict]):
        return self.add(KpiRow(kpis))

    def source(self, text: str, *, prefix: str = "Source: "):
        return self.add(Source(text, prefix=prefix))

    def note(self, text: str):
        return self.add(Note(text))

    def html(self, markup: str):
        return self.add(RawHTML(markup))

    def body_markdown(self, ctx: RenderContext) -> str:
        return "\n\n".join(item.to_markdown(ctx) for item in self.items)


class Column(Container):
    """One column of a :class:`Columns` block."""


class Columns(Component):
    """Columns side by side; ``widths`` are percentages, one per column."""

    def __init__(self, *columns: Column, widths: Sequence[float] | None = None):
        if len(columns) < 2:
            raise ValueError("Columns needs at least two columns.")
        widths = [100 / len(columns)] * len(columns) if widths is None else list(widths)
        if len(widths) != len(columns) or any(w <= 0 for w in widths):
            raise ValueError(f"widths needs {len(columns)} positive percentages.")
        self.columns = list(columns)
        self.widths = widths

    def to_markdown(self, ctx):
        parts = [f'::: {{.column width="{w:g}%"}}\n{c.body_markdown(ctx)}\n:::'
                 for c, w in zip(self.columns, self.widths)]
        return ":::: {.columns}\n" + "\n\n".join(parts) + "\n::::"
