from reportlab.lib import colors

def hex_to_rgb01(hex_str: str):
    """Convert a hex color string to an RGB tuple with values between 0 and 1."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def make_diverging_colormap(low_color=(0, 1, 0), mid_color=(1, 1, 1), high_color=(1, 0, 0)):
    """Create a diverging colormap function that maps values to colors.
    
    Parameters
    ----------
    low_color : tuple
        RGB tuple for the low end of the colormap (default is green).
    mid_color : tuple
        RGB tuple for the middle of the colormap (default is white).
    high_color : tuple
        RGB tuple for the high end of the colormap (default is red).

    Returns
    -------
    function
        A function that takes a value, min, max, and optional mid, and returns a reportlab color.
    """

    def _map(val, vmin, vmax, vmid=None):
        if vmax == vmin:  # avoid division by zero
            return colors.Color(*mid_color if mid_color else low_color)

        # default mid = midpoint
        if vmid is None:
            vmid = (vmax + vmin) / 2.0

        if val <= vmid:
            # interpolate low → mid
            t = (val - vmin) / (vmid - vmin) if vmid > vmin else 0
            r = low_color[0] + t * (mid_color[0] - low_color[0])
            g = low_color[1] + t * (mid_color[1] - low_color[1])
            b = low_color[2] + t * (mid_color[2] - low_color[2])
        else:
            # interpolate mid → high
            t = (val - vmid) / (vmax - vmid) if vmax > vmid else 0
            r = mid_color[0] + t * (high_color[0] - mid_color[0])
            g = mid_color[1] + t * (high_color[1] - mid_color[1])
            b = mid_color[2] + t * (high_color[2] - mid_color[2])

        return colors.Color(r, g, b)

    return _map