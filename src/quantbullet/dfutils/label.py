import numpy as np

def get_bins_and_labels( cutoffs, include_inf=True ):
    if include_inf:
        labels = [f"<{cutoffs[0]}"] + [f"{cutoffs[i]}-{cutoffs[i+1]}" for i in range(len(cutoffs)-1)] + [f">{cutoffs[-1]}"]
        bins = [-np.inf] + cutoffs + [np.inf]
    else:
        bins = cutoffs
        labels = [f"{cutoffs[i]}-{cutoffs[i+1]}" for i in range(len(cutoffs)-1)]
    return bins, labels