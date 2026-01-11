# quantbullet/r/r_session.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class RSession:
    ro: object
    pandas2ri: object
    numpy2ri: object
    localconverter: object

_session: Optional[RSession] = None

def get_r() -> RSession:
    """
    Lazily initialize rpy2 + embedded R.
    Importing this module is safe even if R/rpy2 not installed;
    only calling get_r() requires them.
    """
    global _session
    if _session is not None:
        return _session

    # must be set before importing rpy2
    os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

    try:
        from rpy2 import robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.conversion import localconverter
    except Exception as e:
        raise RuntimeError(
            "R backend is not available. Install R + rpy2 and ensure R_HOME/Rscript is configured."
        ) from e

    _session = RSession(ro=ro, pandas2ri=pandas2ri, numpy2ri=numpy2ri, localconverter=localconverter)
    return _session
