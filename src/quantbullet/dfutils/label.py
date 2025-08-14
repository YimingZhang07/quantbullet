import numpy as np

def get_bins_and_labels(cutoffs, include_inf=True, decimal_places=2, 
                        label_style="simple", right_closed=True):
    """
    Parameters
    ----------
    cutoffs : list or np.ndarray
        Cutoff points for bins.
    include_inf : bool
        Whether to include -inf and inf as outer bounds.
    decimal_places : int
        Number of decimal places to round cutoffs.
    label_style : str
        "simple" -> '<3', '3-6', '>6'
        "interval" -> '( ,3]', '(3,6]', etc.
    right_closed : bool
        For 'interval' style, whether the right side is closed (] vs )).
    """
    # Ensure Python list
    if isinstance(cutoffs, np.ndarray):
        cutoffs = cutoffs.tolist()

    # Round cutoffs
    cutoffs = [round(c, decimal_places) for c in cutoffs]

    # Build bins
    if include_inf:
        bins = [-np.inf] + cutoffs + [np.inf]
    else:
        bins = cutoffs

    # Build labels
    labels = []

    if label_style == "simple":
        if include_inf:
            labels.append(f"<{cutoffs[0]}")
        for i in range(len(cutoffs) - 1):
            labels.append(f"{cutoffs[i]}-{cutoffs[i+1]}")
        if include_inf:
            labels.append(f">{cutoffs[-1]}")

    elif label_style == "interval":
        if right_closed:
            left_bracket = '('
            right_bracket = ']'
        else:
            left_bracket = '['
            right_bracket = ')'

        if include_inf:
            labels.append(f"( ,{cutoffs[0]}{right_bracket}")
        for i in range(len(cutoffs) - 1):
            labels.append(f"{left_bracket}{cutoffs[i]},{cutoffs[i+1]}{right_bracket}")
        if include_inf:
            labels.append(f"{left_bracket}{cutoffs[-1]}, )")

    else:
        raise ValueError("label_style must be 'simple' or 'interval'")

    return bins, labels