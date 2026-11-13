"""Slides: a heading plus components, and the preset layouts.

A layout is a :class:`Slide` subclass whose ``layout`` class attribute is the CSS class put on
the slide; the template's SCSS styles it. Register new layouts with :func:`register_layout` so
``SlideDeck.slide(..., layout=<name>)`` can find them.
"""

from __future__ import annotations

from typing import ClassVar, Sequence

from .components import Column, Columns, Container, RenderContext, fenced_div

_LAYOUTS: dict[str, type[Slide]] = {}


def _attributes(ident: str | None, classes: Sequence[str], attrs: dict[str, str]) -> str:
    parts = ([f"#{ident}"] if ident else []) + [f".{c}" for c in classes]
    parts += [f'{k}="{v}"' for k, v in attrs.items()]
    return f"{{{' '.join(parts)}}}" if parts else ""


def _one_line(text: str | None) -> str:
    return " ".join(str(text).split()) if text else ""


class Slide(Container):
    """One slide: a heading and a list of components.

    ``classes`` are extra CSS classes, ``background`` a background color, ``notes`` speaker
    notes and ``id`` the slide anchor. Use it as a context manager for readability; the
    deck already holds the slide.
    """

    layout: ClassVar[str | None] = None
    level: ClassVar[int] = 2

    def __init__(self, title: str | None = None, *, classes: Sequence[str] = (),
                 background: str | None = None, notes: str | None = None, id: str | None = None):
        super().__init__()
        self.title = title
        self.classes = list(classes)
        self.background = background
        self.notes = notes
        self.id = id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def speaker_notes(self, text: str):
        """Set the speaker notes (shown in the speaker view, key ``s``)."""
        self.notes = text
        return self

    def heading_markdown(self, extra_classes: Sequence[str] = ()) -> str:
        classes = ([self.layout] if self.layout else []) + self.classes + list(extra_classes)
        attrs = {"background-color": self.background} if self.background else {}
        text = " ".join(p for p in (_one_line(self.title), _attributes(self.id, classes, attrs)) if p)
        return "#" * self.level + " " + text

    def to_markdown(self, ctx: RenderContext, *, extra_classes: Sequence[str] = ()) -> str:
        parts = [self.heading_markdown(extra_classes), self.body_markdown(ctx)]
        if self.notes:
            parts.append(fenced_div(self.notes.strip(), "notes"))
        return "\n\n".join(p for p in parts if p)


def register_layout(name: str):
    """Class decorator: register a :class:`Slide` subclass as layout ``name`` (its CSS class)."""

    def decorate(cls: type[Slide]) -> type[Slide]:
        if not (isinstance(cls, type) and issubclass(cls, Slide)):
            raise TypeError("register_layout decorates Slide subclasses.")
        cls.layout = name
        _LAYOUTS[name] = cls
        return cls

    return decorate


def get_layout(name: str) -> type[Slide]:
    """The slide class registered as ``name``."""
    if name not in _LAYOUTS:
        raise KeyError(f"No slide layout {name!r}; registered: {list_layouts()}.")
    return _LAYOUTS[name]


def list_layouts() -> list[str]:
    return sorted(_LAYOUTS)


# ---- Presets -----------------------------------------------------------------------------------

class ContentSlide(Slide):
    """A standard slide: heading with an accent rule, then content."""


_LAYOUTS["content"] = ContentSlide


class SectionSlide(Slide):
    """A section divider (a level-1 heading) on the theme's primary color.

    ``description`` is a line under the title. ``appendix=True`` uses the muted appendix
    background, and the deck tags the slides after it as appendix pages.
    """

    level = 1

    def __init__(self, title: str, *, description: str | None = None, appendix: bool = False,
                 **kwargs):
        super().__init__(title, **kwargs)
        self.description = description
        self.appendix = appendix
        if appendix:
            self.classes.append("appendix-section")

    def body_markdown(self, ctx):
        parts = [fenced_div(self.description.strip(), "section-desc")] if self.description else []
        rest = super().body_markdown(ctx)
        return "\n\n".join(parts + ([rest] if rest else []))


@register_layout("takeaway")
class TakeawaySlide(Slide):
    """One key message in large type with an accent rule."""

    def __init__(self, message: str, title: str | None = None, **kwargs):
        super().__init__(title, **kwargs)
        self.message = message

    def body_markdown(self, ctx):
        rest = super().body_markdown(ctx)
        return fenced_div(self.message.strip(), "takeaway-message") + (f"\n\n{rest}" if rest else "")


@register_layout("two-col")
class TwoColumnSlide(Slide):
    """Two columns, ``left`` and ``right`` (containers); ``widths`` are percentages.

    Components added to the slide itself go below the columns.
    """

    def __init__(self, title: str | None = None, *, widths: Sequence[float] = (45, 55), **kwargs):
        super().__init__(title, **kwargs)
        self.left = Column()
        self.right = Column()
        self.add(Columns(self.left, self.right, widths=widths))


@register_layout("chart-full")
class ChartSlide(Slide):
    """A small heading and one chart across the full width."""

    def __init__(self, title: str | None = None, fig=None, *, height: int = 700, **kwargs):
        chart_kwargs = {k: kwargs.pop(k) for k in ("fmt", "width", "caption", "dpi", "close",
                                                   "apply_theme", "config") if k in kwargs}
        super().__init__(title, **kwargs)
        if fig is not None:
            from .components import is_plotly_figure

            if is_plotly_figure(fig):
                chart_kwargs["height"] = height
            self.chart(fig, **chart_kwargs)


@register_layout("dense")
class DenseSlide(Slide):
    """Smaller text for data-heavy slides (tables, footnotes)."""


@register_layout("agenda")
class AgendaSlide(Slide):
    """A large numbered list. ``items=None`` lists the deck's non-appendix sections."""

    def __init__(self, title: str | None = "Agenda", items: Sequence[str] | None = None, **kwargs):
        super().__init__(title, **kwargs)
        self.agenda_items = None if items is None else [str(i) for i in items]

    def body_markdown(self, ctx):
        items = self.agenda_items if self.agenda_items is not None else ctx.section_titles
        lines = "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))
        rest = super().body_markdown(ctx)
        return "\n\n".join(p for p in (lines, rest) if p)


class PartialSlide(Slide):
    """Slides written in Markdown, inserted as written (e.g. a template's disclaimer)."""

    def __init__(self, markdown: str):
        super().__init__(None)
        self.markdown_text = markdown

    def to_markdown(self, ctx, *, extra_classes=()):
        return self.markdown_text.strip()
