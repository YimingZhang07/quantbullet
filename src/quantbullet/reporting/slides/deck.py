"""SlideDeck: build a reveal.js deck slide by slide in Python, then render it with Quarto."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from .components import RenderContext
from .quarto import render_qmd
from .slides import (
    AgendaSlide,
    ChartSlide,
    PartialSlide,
    SectionSlide,
    Slide,
    TakeawaySlide,
    TwoColumnSlide,
    get_layout,
)
from .template import SlideTemplate, get_template
from .theme import SlideTheme

_QMD_NAME = "index.qmd"


def _front_matter(meta: dict) -> str:
    # YAML block mapping with JSON (YAML flow) values: valid YAML without a YAML library.
    lines = [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in meta.items()
             if value is not None]
    return "---\n" + "\n".join(lines) + "\n---"


class SlideDeck:
    """A slide deck rendered to reveal.js HTML through a Quarto format extension.

    The title slide is made from ``title``, ``subtitle``, ``author`` (one name or several),
    ``date`` and ``classification`` (a small tag such as "Internal"). ``template`` is a
    registered template name or a :class:`SlideTemplate`; ``theme`` overrides the template's
    default :class:`SlideTheme`. ``footer`` is the text at the bottom of content slides;
    ``logo`` shows the template logo (``False`` hides it). ``number_sections`` labels section
    slides 01, 02, ... ``options`` are further reveal.js format options passed to Quarto
    (for example ``{"slide-number": False}``).

    Slides are added in order with :meth:`slide`, :meth:`section`, :meth:`takeaway`,
    :meth:`two_col`, :meth:`chart_slide`, :meth:`agenda`, :meth:`include` or :meth:`add`;
    :meth:`render` writes the HTML.
    """

    def __init__(self, title: str, *, subtitle: str | None = None,
                 author: str | Sequence[str] | None = None,
                 date: str | dt.date | None = None, classification: str | None = None,
                 template: str | SlideTemplate = "default", theme: SlideTheme | None = None,
                 footer: str | None = None, logo: bool = True, number_sections: bool = False,
                 options: dict | None = None):
        self.template = get_template(template) if isinstance(template, str) else template
        self.theme = self.template.default_theme if theme is None else theme
        self.meta: dict = {}
        self.title_slide(title=title, subtitle=subtitle, author=author, date=date,
                         classification=classification)
        self.footer = footer
        self.logo = logo
        self.number_sections = number_sections
        self.options = dict(options or {})
        self.slides: list[Slide] = []

    # ---- building --------------------------------------------------------------------------

    def title_slide(self, **fields) -> SlideDeck:
        """Set title-slide fields: ``title``, ``subtitle``, ``author``, ``date``, ``classification``."""
        allowed = {"title", "subtitle", "author", "date", "classification"}
        unknown = set(fields) - allowed
        if unknown:
            raise TypeError(f"Unknown title slide fields {sorted(unknown)}; use {sorted(allowed)}.")
        for key, value in fields.items():
            if key == "author" and isinstance(value, str):
                value = [value]
            elif key == "author" and value is not None:
                value = [str(v) for v in value]
            elif key == "date" and isinstance(value, dt.date):
                value = value.isoformat()
            self.meta[key] = value
        return self

    def add(self, slide: Slide) -> Slide:
        """Append any :class:`Slide` and return it."""
        if not isinstance(slide, Slide):
            raise TypeError(f"Expected a Slide; got {type(slide).__name__}.")
        self.slides.append(slide)
        return slide

    def slide(self, title: str | None = None, layout: str = "content", **kwargs) -> Slide:
        """Append a slide of a registered ``layout`` ("content", "dense", "two-col", ...).

        ``kwargs`` go to the layout's class, e.g. ``widths`` for "two-col".
        """
        return self.add(get_layout(layout)(title=title, **kwargs))

    def section(self, title: str, *, description: str | None = None,
                appendix: bool = False, **kwargs) -> SectionSlide:
        """Append a section divider; ``appendix=True`` marks the following slides as appendix."""
        return self.add(SectionSlide(title, description=description, appendix=appendix, **kwargs))

    def takeaway(self, message: str, title: str | None = None, **kwargs) -> TakeawaySlide:
        """Append a slide with one key message."""
        return self.add(TakeawaySlide(message, title, **kwargs))

    def two_col(self, title: str | None = None, *, widths: Sequence[float] = (45, 55),
                **kwargs) -> TwoColumnSlide:
        """Append a two-column slide; fill ``.left`` and ``.right``."""
        return self.add(TwoColumnSlide(title, widths=widths, **kwargs))

    def chart_slide(self, title: str | None, fig, **kwargs) -> ChartSlide:
        """Append a slide with one full-width chart (matplotlib or Plotly)."""
        return self.add(ChartSlide(title, fig, **kwargs))

    def agenda(self, items: Sequence[str] | None = None, title: str = "Agenda") -> AgendaSlide:
        """Append an agenda; ``items=None`` lists the non-appendix sections."""
        return self.add(AgendaSlide(title, items))

    def include(self, partial: str) -> PartialSlide:
        """Append the template partial ``partial`` (a Markdown file in its ``partials/``)."""
        return self.add(PartialSlide(self.template.partial(partial)))

    def disclaimer(self) -> PartialSlide:
        """Append the template's disclaimer slide(s)."""
        return self.include("disclaimer")

    # ---- output ----------------------------------------------------------------------------

    def _extension_path(self, relative: str | None) -> str | None:
        if not relative:
            return None
        return f"_extensions/{self.template.extension}/{relative}"

    def _front_matter_dict(self, embed: bool) -> dict:
        fmt = {"footer": self.footer, "embed-resources": True if embed else None}
        if self.logo:
            fmt["logo"] = self._extension_path(self.template.logo)
        fmt.update(self.options)
        meta = {key: self.meta.get(key) for key in
                ("title", "subtitle", "author", "date", "classification")}
        meta.update({
            "qb-title-background": self.theme.primary,
            "qb-title-logo": self._extension_path(self.template.title_logo) if self.logo else None,
            "qb-section-background": self.theme.primary,
            "qb-appendix-background": self.theme.muted,
            "qb-number-sections": self.number_sections,
            "format": {self.template.format_name: {k: v for k, v in fmt.items() if v is not None}},
        })
        if self.theme.webfont_url:
            meta["header-includes"] = (f'```{{=html}}\n<link rel="stylesheet" '
                                       f'href="{self.theme.webfont_url}">\n```')
        return meta

    def to_qmd(self, ctx: RenderContext | None = None, *, embed: bool = False) -> str:
        """The deck as Quarto Markdown. Without ``ctx`` no asset files are written.

        ``embed=True`` asks Quarto for one self-contained HTML file.
        """
        if ctx is None:
            ctx = RenderContext(assets_dir=None, theme=self.theme)
        ctx.section_titles = [s.title for s in self.slides
                              if isinstance(s, SectionSlide) and not s.appendix]
        blocks = []
        in_appendix = False
        for slide in self.slides:
            if isinstance(slide, SectionSlide):
                in_appendix = slide.appendix
            extra = ["appendix"] if in_appendix and slide.level == 2 else []
            blocks.append(slide.to_markdown(ctx, extra_classes=extra))
        return _front_matter(self._front_matter_dict(embed)) +"\n\n" + "\n\n".join(blocks) + "\n"

    def build(self, build_dir: str | Path, *, embed: bool = False) -> Path:
        """Write the Quarto project (``index.qmd``, the extension, assets) to ``build_dir``.

        Returns the path of ``index.qmd``.
        """
        build_dir = Path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        assets = build_dir / "assets"
        if assets.exists():
            shutil.rmtree(assets)
        self.template.install(build_dir, self.theme)
        # A project file stops Quarto from treating a parent folder as the project.
        (build_dir / "_quarto.yml").write_text("project:\n  type: default\n", encoding="utf-8")
        qmd = build_dir / _QMD_NAME
        ctx = RenderContext(assets_dir=assets, theme=self.theme)
        qmd.write_text(self.to_qmd(ctx, embed=embed), encoding="utf-8")
        return qmd

    def render(self, output: str | Path, *, embed: bool = True, keep_build: bool = False,
               quarto: str | Path | None = None, quiet: bool = True) -> Path:
        """Render the deck and return the HTML path.

        ``embed=True`` writes one self-contained HTML file to ``output`` (charts, fonts and
        images inside, opens offline). ``embed=False`` treats ``output`` as a folder and leaves
        the whole Quarto project there, with ``index.html``. ``keep_build=True`` keeps the
        build folder of an embedded render beside ``output`` as ``<stem>_build/`` for debugging.
        ``quarto`` is the Quarto executable (default: ``QUARTO_PATH``, then PATH).
        """
        output = Path(output)
        if not embed:
            return render_qmd(self.build(output), to=self.template.format_name, quarto=quarto,
                              quiet=quiet)

        output.parent.mkdir(parents=True, exist_ok=True)
        if keep_build:
            build_dir = output.parent / f"{output.stem}_build"
            if build_dir.exists():
                shutil.rmtree(build_dir)
        else:
            build_dir = Path(tempfile.mkdtemp(prefix="qb_slides_"))
        try:
            html = render_qmd(self.build(build_dir, embed=True), to=self.template.format_name, quarto=quarto,
                              quiet=quiet)
            shutil.copyfile(html, output)
        finally:
            if not keep_build:
                shutil.rmtree(build_dir, ignore_errors=True)
        return output
