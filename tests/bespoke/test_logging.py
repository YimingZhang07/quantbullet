import unittest
import logging
from quantbullet.log_config import setup_logger
from .test_module import test_f

# # this is a relative import, so we should use package_name.test_module name to reference the __name__ in test_module.py
# test_module_logger = logging.getLogger("bespoke.test_module")

# # Both logger level and handler level should be set to DEBUG to show messagesin the console
# test_module_logger.setLevel(logging.DEBUG)

# if not test_module_logger.hasHandlers():
#     handler = logging.StreamHandler()
#     handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(levelname)-8s %(name)s: %(message)s")
#     handler.setFormatter(formatter)
#     test_module_logger.addHandler(handler)
#     test_module_logger.propagate = False

# updated the setup_logger to include force argument to replicate the above behavior

logger = setup_logger("bespoke.test_module", level=logging.DEBUG, force=True)
logger = setup_logger(__name__, level=logging.WARNING)

class TestLogging(unittest.TestCase):
    def test_logging(self):
        logger.debug("This is a debug message from the test_logging.")
        logger.info("This is a test function in the test_logging.")
        logger.warning("This is a warning message from the test_logging.")
        test_f()