"""Slide decks built in Python and rendered to reveal.js HTML by Quarto.

Build a :class:`SlideDeck` slide by slide, then call :meth:`SlideDeck.render`. Charts and
tables are made in this Python process; Quarto only converts the generated Markdown, so no
Jupyter kernel is involved. Quarto itself must be installed: it is found through the
``QUARTO_PATH`` environment variable or on PATH (see :func:`find_quarto`). Plotly charts need
the ``slides`` extra (``plotly``).

>>> deck = SlideDeck("Monthly Monitor", author="Research", date="2026-10-31")
>>> deck.section("Overview")
>>> with deck.slide("Key numbers") as s:
...     s.kpis([dict(value="12.4", label="1M CPR", delta=2.1)])
>>> deck.render("monitor.html")
"""

from .components import (
    Bullets,
    Column,
    Columns,
    Component,
    Container,
    Figure,
    Image,
    Kpi,
    KpiRow,
    Markdown,
    Note,
    PlotlyFigure,
    RawHTML,
    RenderContext,
    Source,
    Table,
    Text,
)
from .deck import SlideDeck
from .quarto import find_quarto, quarto_version, render_qmd
from .slides import (
    AgendaSlide,
    ChartSlide,
    ContentSlide,
    DenseSlide,
    PartialSlide,
    SectionSlide,
    Slide,
    TakeawaySlide,
    TwoColumnSlide,
    get_layout,
    list_layouts,
    register_layout,
)
from .template import (
    SlideTemplate,
    export_extension,
    export_template,
    get_template,
    list_templates,
    register_template,
)
from .theme import DEFAULT_THEME, SlideTheme

__all__ = [
    "SlideDeck", "SlideTheme", "DEFAULT_THEME", "SlideTemplate",
    "register_template", "get_template", "list_templates", "export_template", "export_extension",
    "Slide", "ContentSlide", "SectionSlide", "TakeawaySlide", "TwoColumnSlide", "ChartSlide",
    "DenseSlide", "AgendaSlide", "PartialSlide", "register_layout", "get_layout", "list_layouts",
    "Component", "Container", "Column", "Columns", "Markdown", "Text", "Bullets", "Figure",
    "PlotlyFigure", "Image", "Table", "Kpi", "KpiRow", "Source", "Note", "RawHTML",
    "RenderContext", "find_quarto", "quarto_version", "render_qmd",
]
