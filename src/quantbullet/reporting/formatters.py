import numpy as np
import pandas as pd

def round_errors( x, decimals=2 ):
    rounded = round(x, decimals)
    return abs( rounded - x ) / abs( x ) if x != 0 else 0

def default_number_formatter(val, decimals=2, comma=False):
    """Format a number with specified digits and optional comma as thousand separator."""
    if pd.isna(val):
        return ""
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
