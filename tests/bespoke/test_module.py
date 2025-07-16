import logging
from quantbullet.log_config import setup_logger

# logger = logging.getLogger( __name__ )
logger = setup_logger(__name__)

def test_f():
    logger.debug("This is a debug message from the test_module.")
    logger.info("This is a test function in the test_module.")
    logger.warning("This is a warning message from the test_module.")