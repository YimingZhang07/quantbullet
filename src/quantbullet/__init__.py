# read version from installed package
from importlib.metadata import version
from .log_config import setup_logger, set_package_log_level
from .global_utils import set_figsize
from .model import *
from .utils import *
from .research import *

# Package details
__version__ = version("quantbullet")


def display_package_info(info_dict):
    """
    Display package information in a geeky bordered box format.

    Args:
        info_dict (dict): Dictionary containing package details.
    """
    # ANSI escape codes for color
    GREEN = "\033[92m"
    # YELLOW = "\033[93m"
    RESET = "\033[0m"

    # Calculate the width of the box based on the content
    max_key_width = max(len(str(k)) for k in info_dict.keys())
    width = (
        max_key_width + max(len(str(v)) for v in info_dict.values()) + 6
    )  # 6 for ": ", spaces, and padding

    # Create the box
    top_border = GREEN + "+" + "-" * width + "+" + RESET
    middle = "|  {key:<{key_width}} : {value:<{value_width}} |"
    bottom_border = top_border

    # Print the box with content
    print(top_border)
    for key, value in info_dict.items():
        value_width = width - max_key_width - 6  # 6 for ": ", spaces, and padding
        print(
            middle.format(
                key=key, value=value, key_width=max_key_width, value_width=value_width
            )
        )
    print(bottom_border)


info = {
    "Package": "quantbullet",
    "Author" : "Yiming Zhang",
    "Version": __version__,
    "Note"   : "BETA version",
}
# display_package_info(info)
