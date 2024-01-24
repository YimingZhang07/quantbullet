"""
Centralized logging configuration for the package.
"""
import logging

LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


def setup_logger(name):
    """Setup a logger with a given name."""
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        # formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s",
        #                               datefmt="%m-%d %H:%M:%S")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                datefmt="%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # this is just the default log level, it can be changed later
        logger.setLevel(logging.INFO)

    return logger


def set_package_log_level(level='WARNING'):
    """Set the log level for all loggers in the package."""
    level = LEVEL_MAP.get(level.upper(), logging.WARNING)
    top_level_name = __name__.split('.', maxsplit=1)[0]
    for logger_name, logger_instance in logging.root.manager.loggerDict.items():
        if logger_name.startswith(top_level_name) and\
                isinstance(logger_instance, logging.Logger):
            logger_instance.setLevel(level)


# package logger
pkg_logger = setup_logger(__name__)
