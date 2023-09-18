# read version from installed package
from importlib.metadata import version
__version__ = version("quantbullet")

from .log_config import setup_logger, set_package_log_level