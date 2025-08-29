import numpy as np

def round_errors( x, decimals=2 ):
    rounded = round(x, decimals)
    return abs( rounded - x ) / abs( x ) if x != 0 else 0

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
    formatted = [number2string(v, **kwargs) for v in values]
    return "[" + ", ".join(formatted) + "]"
