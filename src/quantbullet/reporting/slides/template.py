"""Slide templates: a folder holding a Quarto reveal.js format extension, plus a registry.

A template folder contains

- ``_extension.yml``: the Quarto format extension (reveal.js options, theme, partials, filters);
- ``theme.scss``: the SCSS theme; :meth:`SlideTemplate.install` writes the
  :class:`~quantbullet.reporting.slides.SlideTheme` variables into its defaults section;
- ``fonts.html``: a placeholder that :meth:`SlideTemplate.install` replaces with the embedded fonts;
- ``partials/*.md``: fixed slides such as a disclaimer, inserted with ``SlideDeck.include``;
- ``template.json`` (optional): ``name``, ``description``, ``extension`` (the Quarto extension
  name, default ``qb-<name>``), ``logo`` and ``title_logo`` (paths inside the folder) and
  ``theme`` (token overrides for this template's default :class:`SlideTheme`).

Copy the built-in template with :func:`export_template`, edit it, then load it with
:meth:`SlideTemplate.from_dir` and :func:`register_template`.
"""

from __future__ import annotations

import base64
import importlib.resources
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .theme import DEFAULT_THEME, SlideTheme

_DEFAULTS_MARKER = "/*-- scss:defaults --*/"
_META_FILE = "template.json"


@dataclass(frozen=True)
class SlideTemplate:
    """A slide template folder; see the module docstring for its layout."""

    name: str
    root: Path
    description: str = ""
    extension: str = ""
    default_theme: SlideTheme = DEFAULT_THEME
    logo: str | None = None
    title_logo: str | None = None
    partials: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        if not self.extension:
            object.__setattr__(self, "extension", f"qb-{self.name}")

    @classmethod
    def from_dir(cls, path: str | Path, theme: SlideTheme | None = None) -> SlideTemplate:
        """Load a template folder. ``theme`` replaces the template's default theme."""
        root = Path(path).resolve()
        for required in ("_extension.yml", "theme.scss"):
            if not (root / required).is_file():
                raise FileNotFoundError(f"{root} is not a slide template: {required} is missing.")
        meta = {}
        if (root / _META_FILE).is_file():
            meta = json.loads((root / _META_FILE).read_text(encoding="utf-8"))
        if theme is None:
            theme = DEFAULT_THEME.replace(**meta.get("theme", {}))
        partials = {p.stem: p for p in sorted((root / "partials").glob("*.md"))}
        for key in ("logo", "title_logo"):
            if meta.get(key) and not (root / meta[key]).is_file():
                raise FileNotFoundError(f"{key} {meta[key]!r} is not a file in {root}.")
        return cls(name=meta.get("name", root.name), root=root,
                   description=meta.get("description", ""), extension=meta.get("extension", ""),
                   default_theme=theme, logo=meta.get("logo"), title_logo=meta.get("title_logo"),
                   partials=partials)

    @property
    def format_name(self) -> str:
        """The Quarto format a deck declares, e.g. ``qb-default-revealjs``."""
        return f"{self.extension}-revealjs"

    def extension_dir(self, project_dir: str | Path) -> Path:
        return Path(project_dir) / "_extensions" / self.extension

    def partial(self, name: str) -> str:
        """Markdown text of the partial ``name`` (a file in ``partials/``)."""
        if name not in self.partials:
            raise KeyError(f"Template {self.name!r} has no partial {name!r}; "
                           f"it has {sorted(self.partials)}.")
        return self.partials[name].read_text(encoding="utf-8")

    def install(self, project_dir: str | Path, theme: SlideTheme | None = None) -> Path:
        """Copy the template into ``project_dir/_extensions/<extension>/`` styled by ``theme``.

        Returns the extension folder.
        """
        theme = self.default_theme if theme is None else theme
        dest = self.extension_dir(project_dir)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.root, dest)

        scss_path = dest / "theme.scss"
        scss = scss_path.read_text(encoding="utf-8")
        if _DEFAULTS_MARKER not in scss:
            raise ValueError(f"{self.root / 'theme.scss'} has no '{_DEFAULTS_MARKER}' section.")
        tokens = "// Written by SlideTemplate.install from the deck's SlideTheme\n" + theme.to_scss()
        scss_path.write_text(scss.replace(_DEFAULTS_MARKER, f"{_DEFAULTS_MARKER}\n{tokens}", 1),
                             encoding="utf-8")
        (dest / "fonts.html").write_text(_font_faces(theme), encoding="utf-8")
        return dest


def _font_faces(theme: SlideTheme) -> str:
    """A <style> block embedding the theme's bundled font files, or a comment when none."""
    fonts = theme.embedded_fonts()
    if not fonts:
        return "<!-- no bundled font for this theme -->\n"
    styles = {"Regular": ("normal", "normal"), "Bold": ("bold", "normal"),
              "Italic": ("normal", "italic")}
    rules = []
    for family, key, path in fonts:
        weight, style = styles[key]
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        rules.append(f'@font-face {{ font-family: "{family}"; font-weight: {weight}; '
                     f'font-style: {style}; src: url(data:font/ttf;base64,{data}) format("truetype"); }}')
    return "<style>\n" + "\n".join(rules) + "\n</style>\n"


# ---- registry ------------------------------------------------------------------------------

_REGISTRY: dict[str, SlideTemplate] = {}


def _builtin_dir(name: str) -> Path:
    return Path(str(importlib.resources.files("quantbullet.reporting.slides") / "templates" / name))


BUILTIN_TEMPLATES = ("default", "editorial")


def _ensure_builtins() -> None:
    for name in BUILTIN_TEMPLATES:
        if name not in _REGISTRY:
            _REGISTRY[name] = SlideTemplate.from_dir(_builtin_dir(name))


def register_template(template: SlideTemplate, *, replace: bool = False) -> None:
    """Make ``template`` available to ``SlideDeck(template=<name>)``."""
    _ensure_builtins()
    if template.name in _REGISTRY and not replace:
        raise ValueError(f"A template named {template.name!r} is already registered; "
                         "pass replace=True to swap it.")
    _REGISTRY[template.name] = template


def get_template(name: str) -> SlideTemplate:
    """Return the registered template ``name``."""
    _ensure_builtins()
    if name not in _REGISTRY:
        raise KeyError(f"No slide template {name!r}; registered: {list_templates()}.")
    return _REGISTRY[name]


def list_templates() -> list[str]:
    """Names of the registered templates."""
    _ensure_builtins()
    return sorted(_REGISTRY)


def export_template(name: str, dest: str | Path) -> Path:
    """Copy the template ``name`` to the new folder ``dest`` to start a custom template.

    The copy is renamed after ``dest``'s folder name (``template.json``'s ``name``, and the
    extension ``qb-<folder>``). Returns ``dest``.
    """
    source = get_template(name)
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists.")
    shutil.copytree(source.root, dest)
    meta_path = dest / _META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta.update(name=dest.name, extension=f"qb-{dest.name}")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return dest


def export_extension(name: str, project_dir: str | Path, theme: SlideTheme | None = None) -> Path:
    """Install the template ``name`` into a Quarto project for hand-written ``.qmd`` decks.

    Afterwards a deck in ``project_dir`` declares ``format: <extension>-revealjs``.
    Returns the extension folder.
    """
    return get_template(name).install(project_dir, theme)
