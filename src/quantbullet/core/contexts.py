import logging
from contextlib import contextmanager

@contextmanager
def suppress_logging(level=logging.ERROR):
    """
    Temporarily suppress all logging messages at or below a given level.

    Args:
        level: The maximum log level to suppress (default: logging.ERROR).
    """
    logger = logging.getLogger()
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(level + 1)  # suppress `level` and below
    try:
        yield
    finally:
        logger.setLevel(previous_level)
