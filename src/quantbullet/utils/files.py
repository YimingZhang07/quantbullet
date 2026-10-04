"""Small file helpers shared by workflows."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import tempfile


def file_sha256(path: Path) -> str:
    """Hash a file in 8 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def temporary_output(target: Path) -> Iterator[Path]:
    """Write beside ``target`` and replace it only after the caller succeeds."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{target.stem}-", suffix=target.suffix, dir=target.parent)
    os.close(handle)
    path = Path(name)
    try:
        yield path
        path.replace(target)
    finally:
        path.unlink(missing_ok=True)


def expand_env_path(value: str, *, base: Path) -> Path:
    """Expand ``${VAR}`` and resolve a path relative to ``base``."""
    def variable(match: re.Match[str]) -> str:
        if not os.environ.get(match[1]):
            raise ValueError(f"Set environment variable {match[1]}")
        return os.environ[match[1]]

    expanded = Path(re.sub(r"\$\{([A-Za-z_]\w*)\}", variable, value)).expanduser()
    return (expanded if expanded.is_absolute() else Path(base) / expanded).resolve()
