# quantbullet/r/r_session.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


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
        Path to the R installation root (e.g. ``C:/Program Files/R/R-4.4.0``)
        or to the ``Rscript`` executable.  When provided, ``R_HOME`` is set
        before rpy2 is imported so that rpy2 finds the correct R.
    use_renv : bool
        When *True* (default), ``renv::load()`` is called on startup to
        activate the project-local renv library.  Set to *False* on machines
        where all R packages are already installed system-wide.
    """
    r_home: Optional[str] = None
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
        use_renv: bool = True,
    ) -> None:
        """Store backend options.  Must be called before :meth:`get_session`."""
        if self._session is not None:
            raise RuntimeError(
                "Cannot configure R after the session is already initialised. "
                "Call configure_r() before the first get_r()."
            )
        self._config = RConfig(r_home=r_home, use_renv=use_renv)

    def get_session(self) -> RSession:
        """Return the live session, booting R on first call."""
        if self._session is not None:
            return self._session

        if self._config.r_home is not None:
            resolved = _resolve_r_home(self._config.r_home)
            os.environ["R_HOME"] = resolved
            r_bin = Path(resolved) / "bin"
            r_bin_x64 = r_bin / "x64"
            extra = os.pathsep.join(
                str(p) for p in (r_bin_x64, r_bin) if p.is_dir()
            )
            if extra:
                os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")

        # must be set before importing rpy2
        os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
        os.environ["RENV_CONFIG_AUTOLOADER_ENABLED"] = "false"

        try:
            from rpy2 import robjects as ro
            from rpy2.robjects import pandas2ri, numpy2ri
            from rpy2.robjects.conversion import localconverter
        except Exception as e:
            raise RuntimeError(
                "R backend is not available. "
                "Install R + rpy2 and ensure R_HOME/Rscript is configured."
            ) from e

        if self._config.r_home is not None:
            ro.r('.libPaths(c(file.path(R.home(), "library")))')

        if self._config.use_renv:
            ro.r("renv::load()")

        self._session = RSession(
            ro=ro,
            pandas2ri=pandas2ri,
            numpy2ri=numpy2ri,
            localconverter=localconverter,
        )
        return self._session


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_r_home(path_str: str) -> str:
    """Derive ``R_HOME`` from *path_str*.

    Accepts either the R installation root directory or a path to the
    ``Rscript`` / ``R`` executable and walks up to the installation root.
    """
    p = Path(path_str).resolve()

    if p.is_file():
        # e.g. .../R-4.4.0/bin/x64/Rscript.exe  ->  .../R-4.4.0
        #  or  .../R-4.4.0/bin/Rscript.exe        ->  .../R-4.4.0
        candidate = p.parent
        while candidate != candidate.parent:
            if (candidate / "bin").is_dir() and candidate.name != "bin":
                return str(candidate)
            candidate = candidate.parent
        raise ValueError(
            f"Cannot derive R_HOME from executable path: {path_str}"
        )

    if p.is_dir():
        if (p / "bin").is_dir():
            return str(p)
        raise ValueError(
            f"Directory does not look like an R installation "
            f"(no bin/ subdir): {path_str}"
        )

    raise FileNotFoundError(f"Path does not exist: {path_str}")


# ---------------------------------------------------------------------------
# Public module-level API (delegates to the singleton)
# ---------------------------------------------------------------------------

def configure_r(
    r_home: Optional[str] = None,
    use_renv: bool = True,
) -> None:
    """Set R backend options **before** the first ``get_r()`` call.

    The two parameters are independent and combine as follows:

    +------------+------------+------------------------------------------------+
    | r_home     | use_renv   | Behaviour                                      |
    +============+============+================================================+
    | None       | True       | **Default / legacy.** rpy2 discovers R via     |
    |            |            | ``R_HOME`` or ``PATH``; ``renv::load()`` is    |
    |            |            | called to activate the project library.         |
    +------------+------------+------------------------------------------------+
    | None       | False      | System R on PATH with packages installed        |
    |            |            | globally; renv is skipped.                      |
    +------------+------------+------------------------------------------------+
    | set        | False      | **New-machine shortcut.** Point to a specific   |
    |            |            | R installation and skip renv entirely.           |
    +------------+------------+------------------------------------------------+
    | set        | True       | Custom R path *with* renv (unusual; only if the |
    |            |            | renv project lives under a non-default R).       |
    +------------+------------+------------------------------------------------+

    Parameters
    ----------
    r_home : str, optional
        Path to R installation root (e.g. ``C:/Program Files/R/R-4.4.0``)
        or to the ``Rscript`` / ``R`` executable.  When provided, ``R_HOME``
        is set before rpy2 is imported so that rpy2 finds the correct R.
        When *None*, rpy2 discovers R through its normal mechanism.
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

    Use system R on PATH without renv:

    >>> configure_r(use_renv=False)
    """
    RSessionManager.instance().configure(r_home=r_home, use_renv=use_renv)


def get_r() -> RSession:
    """Lazily initialize rpy2 + embedded R.

    Importing this module is safe even if R / rpy2 is not installed;
    only calling ``get_r()`` requires them.

    If :func:`configure_r` was called beforehand, its settings are
    respected (custom ``R_HOME``, renv on/off).  Otherwise the legacy
    behaviour (renv activated) is preserved.
    """
    return RSessionManager.instance().get_session()
