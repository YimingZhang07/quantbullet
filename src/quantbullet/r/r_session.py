# quantbullet/r/r_session.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import os
import sys

# On Windows, rpy2 may briefly try API mode before falling back to ABI.
# Setting this early avoids noisy fallback messages while still allowing
# callers to override the mode explicitly before importing this module.
if sys.platform == "win32":
    os.environ.setdefault("RPY2_CFFI_MODE", "ABI")


@dataclass
class RSession:
    ro: object
    pandas2ri: object
    numpy2ri: object
    localconverter: object


@dataclass
class RConfig:
    """Configuration for the R backend.

    Parameters
    ----------
    r_home : str, optional
        Path to the R installation root (e.g. ``C:/Program Files/R/R-4.4.0``).
        When provided, ``R_HOME`` is set before rpy2 is imported so that rpy2
        finds the correct R.
    lib_paths : sequence of str, optional
        Explicit R library directories to use for package lookup. These are
        validated and applied via ``.libPaths()`` before any project packages
        are loaded.
    include_r_home_library : bool
        When *True* (default), prepend ``<R_HOME>/library`` to explicit
        library paths so the selected R's base/recommended packages stay
        visible.
    use_renv : bool
        When *True* (default), ``renv::load()`` is called on startup to
        activate the project-local renv library.  Set to *False* on machines
        where all R packages are already installed system-wide.
    """
    r_home: Optional[str] = None
    lib_paths: Optional[tuple[str, ...]] = None
    include_r_home_library: bool = True
    use_renv: bool = True


class RSessionManager:
    """Singleton that owns the R configuration and the live session.

    rpy2 only supports a single embedded R process per Python process,
    so this is genuinely process-wide state.  Encapsulating it in a class
    (rather than bare module globals) keeps the lifecycle explicit and
    makes it possible to reset state in tests.
    """

    _instance: Optional[RSessionManager] = None

    def __init__(self) -> None:
        self._config = RConfig()
        self._session: Optional[RSession] = None
        self._sourced_files: set[str] = set()

    @classmethod
    def instance(cls) -> RSessionManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton (for tests). Does **not** shut down R."""
        cls._instance = None

    @property
    def is_initialised(self) -> bool:
        return self._session is not None

    def configure(
        self,
        r_home: Optional[str] = None,
        lib_paths: Optional[Sequence[str]] = None,
        include_r_home_library: bool = True,
        use_renv: bool = True,
    ) -> None:
        """Store backend options.  Must be called before :meth:`get_session`."""
        import warnings

        new_config = RConfig(
            r_home=r_home,
            lib_paths=tuple(lib_paths) if lib_paths is not None else None,
            include_r_home_library=include_r_home_library,
            use_renv=use_renv,
        )
        if self._session is not None:
            if new_config == self._config:
                return
            warnings.warn(
                "R session is already initialised; ignoring new configuration. "
                "Restart the Python process to apply different R settings.",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        self._config = new_config

    def get_session(self) -> RSession:
        """Return the live session, booting R on first call."""
        if self._session is not None:
            return self._session

        resolved_r_home: Optional[str] = None
        if self._config.r_home is not None:
            resolved_r_home = _validate_r_home(self._config.r_home)
            os.environ["R_HOME"] = resolved_r_home
            r_bin = Path(resolved_r_home) / "bin"
            r_bin_x64 = r_bin / "x64"
            extra = os.pathsep.join(
                str(p) for p in (r_bin_x64, r_bin) if p.is_dir()
            )
            if extra:
                os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")

        # must be set before importing rpy2
        os.environ["RENV_CONFIG_AUTOLOADER_ENABLED"] = "false"

        try:
            from rpy2 import robjects as ro
            from rpy2.robjects import pandas2ri, numpy2ri
            from rpy2.robjects.conversion import localconverter
        except Exception as e:
            raise RuntimeError(
                "R backend is not available. "
                "Install R + rpy2 and ensure R_HOME or PATH points to a working R installation."
            ) from e

        if self._config.r_home is not None or self._config.lib_paths is not None:
            runtime_r_home = resolved_r_home or str(ro.r("R.home()")[0])
            lib_paths = _resolve_lib_paths(
                r_home=runtime_r_home,
                lib_paths=self._config.lib_paths,
                include_r_home_library=self._config.include_r_home_library,
            )
            if lib_paths:
                ro.r[".libPaths"](ro.StrVector(lib_paths))

        if self._config.use_renv:
            try:
                ro.r("renv::load()")
            except Exception as e:
                raise RuntimeError(
                    "Failed to load renv. Ensure the renv package is installed in one "
                    "of the configured lib_paths, or disable renv with use_renv=False."
                ) from e

        self._session = RSession(
            ro=ro,
            pandas2ri=pandas2ri,
            numpy2ri=numpy2ri,
            localconverter=localconverter,
        )
        return self._session

    def source_file(self, path: str | Path, *, force: bool = False) -> None:
        """Source an R file at most once per Python process by default."""
        resolved = Path(path).resolve()
        key = str(resolved)
        if not force and key in self._sourced_files:
            return

        session = self.get_session()
        session.ro.r["source"](resolved.as_posix())
        self._sourced_files.add(key)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_r_home(path_str: str) -> str:
    """Validate that *path_str* is an R installation root."""
    p = Path(path_str).expanduser()

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path_str}")

    if p.is_file():
        raise ValueError(
            "r_home must be the R installation root (the directory that contains "
            f"bin/), not the R or Rscript executable: {path_str}"
        )

    p = p.resolve()
    if not (p / "bin").is_dir():
        raise ValueError(
            "r_home must point to the R installation root and contain a bin/ "
            f"subdirectory: {path_str}"
        )

    return str(p)


def _validate_lib_path(path_str: str) -> str:
    """Validate that *path_str* is an existing R library directory."""
    p = Path(path_str).expanduser()

    if not p.exists():
        raise FileNotFoundError(f"Library path does not exist: {path_str}")

    if not p.is_dir():
        raise ValueError(f"Library path must be a directory: {path_str}")

    return str(p.resolve())


def _resolve_lib_paths(
    r_home: str,
    lib_paths: Optional[Sequence[str]],
    include_r_home_library: bool,
) -> list[str]:
    """Build the explicit ``.libPaths()`` search list."""
    resolved_paths: list[str] = []

    if include_r_home_library:
        resolved_paths.append(str((Path(r_home) / "library").resolve()))

    if lib_paths is not None:
        resolved_paths.extend(_validate_lib_path(path) for path in lib_paths)

    deduped_paths: list[str] = []
    seen: set[str] = set()
    for path in resolved_paths:
        if path not in seen:
            seen.add(path)
            deduped_paths.append(path)

    return deduped_paths


# ---------------------------------------------------------------------------
# Public module-level API (delegates to the singleton)
# ---------------------------------------------------------------------------

def configure_r(
    r_home: Optional[str] = None,
    lib_paths: Optional[Sequence[str]] = None,
    include_r_home_library: bool = True,
    use_renv: bool = True,
) -> None:
    """Set R backend options **before** the first ``get_r()`` call.

    The parameters combine as follows:

    +------------+------------+------------------------+------------+----------------------+
    | r_home     | lib_paths  | include_r_home_library | use_renv   | Behaviour            |
    +============+============+========================+============+======================+
    | None       | None       | True                   | True/False | Legacy discovery for |
    |            |            |                        |            | both R and libs.     |
    +------------+------------+------------------------+------------+----------------------+
    | set/None   | set        | True                   | True/False | Explicit lib paths   |
    |            |            |                        |            | plus ``R_HOME``      |
    |            |            |                        |            | library.             |
    +------------+------------+------------------------+------------+----------------------+
    | set/None   | set        | False                  | True/False | Only explicit        |
    |            |            |                        |            | ``lib_paths``.       |
    +------------+------------+------------------------+------------+----------------------+
    | set        | None       | True                   | True/False | Use the selected R's |
    |            |            |                        |            | own ``library``      |
    |            |            |                        |            | directory only.      |
    +------------+------------+------------------------+------------+----------------------+

    Parameters
    ----------
    r_home : str, optional
        Path to the R installation root (e.g. ``C:/Program Files/R/R-4.4.0``).
        When provided, ``R_HOME`` is set before rpy2 is imported so that rpy2
        finds the correct R. When *None*, rpy2 discovers R through its normal
        mechanism.
    lib_paths : sequence of str, optional
        Explicit R library directories to apply via ``.libPaths()``. These are
        not inferred from the environment.
    include_r_home_library : bool
        Whether to prepend the selected R installation's ``library``
        directory to ``lib_paths``. Default is *True*.
    use_renv : bool
        Whether to call ``renv::load()`` on startup (default *True*).
        Set to *False* on machines where all R packages are already
        installed system-wide.

    Raises
    ------
    RuntimeError
        If the R session has already been initialised.

    Examples
    --------
    Use a specific R installation without renv:

    >>> from quantbullet.r.r_session import configure_r
    >>> configure_r(r_home=r"C:\\Program Files\\R\\R-4.4.0", use_renv=False)

    Use a specific R installation with an explicit user library:

    >>> configure_r(
    ...     r_home=r"C:\\Program Files\\R\\R-4.4.0",
    ...     lib_paths=[r"C:\\Users\\you\\AppData\\Local\\R\\win-library\\4.4"],
    ... )

    Use system R on PATH without renv:

    >>> configure_r(use_renv=False)
    """
    RSessionManager.instance().configure(
        r_home=r_home,
        lib_paths=lib_paths,
        include_r_home_library=include_r_home_library,
        use_renv=use_renv,
    )


def get_r() -> RSession:
    """Lazily initialize rpy2 + embedded R.

    Importing this module is safe even if R / rpy2 is not installed;
    only calling ``get_r()`` requires them.

    If :func:`configure_r` was called beforehand, its settings are
    respected (custom ``R_HOME``, renv on/off).  Otherwise the legacy
    behaviour (renv activated) is preserved.
    """
    return RSessionManager.instance().get_session()


def source_r_file(path: str | Path, *, force: bool = False) -> None:
    """Source an R file once for the current Python process."""
    RSessionManager.instance().source_file(path, force=force)
