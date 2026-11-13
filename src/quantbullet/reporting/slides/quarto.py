"""Locate the Quarto command-line tool and render a ``.qmd`` file with it."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

MIN_VERSION = (1, 5, 0)

_INSTALL_HINT = (
    "Quarto was not found. Install it from https://quarto.org/docs/get-started/ "
    "(or `winget install Posit.Quarto`), then either put `quarto` on PATH, set the "
    "QUARTO_PATH environment variable to the quarto executable, or pass `quarto=` explicitly."
)


def find_quarto(path: str | Path | None = None) -> Path:
    """Return the Quarto executable.

    Looks at ``path``, then the ``QUARTO_PATH`` environment variable, then ``quarto``
    on PATH, then the binary of the ``quarto_cli`` Python package when installed.
    ``path`` and ``QUARTO_PATH`` may name the executable or the folder holding it.
    """
    for candidate in (path, os.environ.get("QUARTO_PATH")):
        if candidate:
            found = _executable(Path(candidate))
            if found is None:
                raise FileNotFoundError(f"No quarto executable at {candidate!r}.")
            return found
    on_path = shutil.which("quarto")
    if on_path:
        return Path(on_path)
    bundled = _quarto_cli_binary()
    if bundled is not None:
        return bundled
    raise FileNotFoundError(_INSTALL_HINT)


def _executable(path: Path) -> Path | None:
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("quarto.exe", "quarto.cmd", "quarto"):
            for folder in (path, path / "bin"):
                if (folder / name).is_file():
                    return folder / name
    return None


def _quarto_cli_binary() -> Path | None:
    try:
        import quarto_cli  # type: ignore[import-not-found]
    except ImportError:
        return None
    return _executable(Path(quarto_cli.__file__).parent / "bin")


def quarto_version(quarto: str | Path | None = None) -> tuple[int, int, int]:
    """Return the (major, minor, patch) version of Quarto."""
    exe = find_quarto(quarto)
    out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
    if match is None:
        raise RuntimeError(f"Could not read the Quarto version from {out!r}.")
    return tuple(int(part) for part in match.groups())


def render_qmd(qmd: str | Path, *, to: str, quarto: str | Path | None = None,
               quiet: bool = True) -> Path:
    """Render ``qmd`` to the format ``to`` and return the HTML file Quarto wrote.

    The HTML is written beside ``qmd`` with the same stem. A failed render raises
    ``RuntimeError`` carrying Quarto's output.
    """
    exe = find_quarto(quarto)
    version = quarto_version(exe)
    if version < MIN_VERSION:
        raise RuntimeError(f"Quarto {'.'.join(map(str, MIN_VERSION))} or later is needed; found "
                           f"{'.'.join(map(str, version))}.")
    qmd = Path(qmd).resolve()
    cmd = [str(exe), "render", qmd.name, "--to", to]
    if quiet:
        cmd.append("--quiet")
    result = subprocess.run(cmd, cwd=qmd.parent, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"quarto render failed (exit {result.returncode}):\n"
                           f"{result.stderr.strip() or result.stdout.strip()}")
    html = qmd.with_suffix(".html")
    if not html.is_file():
        raise RuntimeError(f"quarto render finished but {html} was not written.")
    return html
