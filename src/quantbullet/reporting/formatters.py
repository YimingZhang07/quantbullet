import numpy as np
import pandas as pd

def round_errors( x, decimals=2 ):
    rounded = round(x, decimals)
    return abs( rounded - x ) / abs( x ) if x != 0 else 0

def flex_number_formatter(val, decimals=2, comma=False, transformer=None, percent=False):
    """Format a number with specified digits and optional comma as thousand separator.
    
    Parameters
    ----------
    val : float or int
        The number to format.
    decimals : int
        Number of decimal places.
    comma : bool
        Whether to include commas as thousand separators.
    transformer : callable, optional
        A function to transform the value before formatting (e.g., scaling).
    """
    if pd.isna(val):
        return ""
    
    if transformer:
        val = transformer(val)

    if percent:
        val *= 100
        return flex_number_formatter(val, decimals=decimals, comma=comma) + '%'

    fmt = f"{{:,.{decimals}f}}" if comma else f"{{:.{decimals}f}}"
    return fmt.format(val)

def number2string(x, tol=1e-9, sigfigs=2):
    """
    Smarter number formatter:
    - Large ints: commas
    - Floats near ints: print as int
    - Floats: try rounding to normal_decimals; if distortion too big, try fewer decimals
              if still too big, fallback to scientific
    - Very small/large floats: scientific
    """
    # Integers
    if np.issubdtype(type(x), np.integer):
        return f"{x:,}" if abs(x) >= 1000 else str(x)

    # Floats
    elif np.issubdtype(type(x), np.floating):
        if x == 0.0:
            return "0"

        # Float looks like int
        if abs(x - round(x)) < tol:
            return f"{int(round(x)):,}"

        absx = abs(x)

        if 1e-4 <= absx < 1e6:
            if round_errors(x, 2) < 1e-3:
                return f"{x:,.2f}"

            if round_errors(x, 4) < 1e-3:
                return f"{x:,.4f}"

        # Otherwise → scientific
        return f"{x:.{sigfigs}e}"

    # Other types
    return str(x)

def numberarray2string(values, **kwargs):
    """
    Format a list/array of numbers into a string representation.
    Example: [1.23, 4.56, 7.89] → "[1.23, 4.56, 7.89]"
    """
    formatted = [number2string(v, **kwargs) for v in values]
    return "[" + ", ".join(formatted) + "]"


def _strip_trailing_zeros(s: str) -> str:
    """Strip trailing zeros and a trailing decimal point, e.g. '1.20' -> '1.2', '3.00' -> '3'."""
    if "." not in s:
        return s
    s = s.rstrip("0").rstrip(".")
    return s


def human_number(
    x,
    *,
    decimals: int = 2,
    decimals_lt1: int = 4,
    decimals_suffix: int = 2,
    small_sci_threshold: float = 1e-2,
    sci_sigfigs: int = 2,
    use_suffixes: bool = True,
    upper: bool = True,
):
    """
    Human-friendly number formatter with deterministic rules:
    - NaN/None -> ""
    - 0 -> "0"
    - |x| < small_sci_threshold -> scientific (e.g. 1.2e-4)
    - small values:
        - |x| < 1 -> fixed with `decimals_lt1` (trim trailing zeros)
        - |x| < 1000 -> fixed with `decimals` (trim trailing zeros)
    - large values (if use_suffixes=True):
        - >= 1e3 -> K, >= 1e6 -> M, >= 1e9 -> B, >= 1e12 -> T (trim trailing zeros)

    Notes
    -----
    - This is meant for readability (dashboard/report labels), not exact numeric round-trip.
    - Negative numbers preserve the sign.
    """
    if x is None or pd.isna(x):
        return ""

    # numpy scalars -> python scalars
    if isinstance(x, (np.generic,)):
        x = x.item()

    try:
        x = float(x)
    except Exception:
        return str(x)

    if x == 0.0:
        return "0"

    sign = "-" if x < 0 else ""
    ax = abs(x)

    # Very small -> scientific
    if ax < float(small_sci_threshold):
        return sign + f"{ax:.{int(sci_sigfigs)}e}"

    # Small/moderate without suffixes
    if not use_suffixes or ax < 1000.0:
        d = int(decimals_lt1) if ax < 1.0 else int(decimals)
        return sign + _strip_trailing_zeros(f"{ax:.{d}f}")

    # Large with suffix
    suffixes = [
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    ]
    for div, suf in suffixes:
        if ax >= div:
            val = ax / div
            s = _strip_trailing_zeros(f"{val:.{int(decimals_suffix)}f}")
            suf = suf if upper else suf.lower()
            return f"{sign}{s}{suf}"

    # Fallback (shouldn't happen due to ax>=1000 check above)
    return sign + _strip_trailing_zeros(f"{ax:.{int(decimals)}f}")