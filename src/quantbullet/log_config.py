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

def setup_logger(name: str, level: int = logging.DEBUG, propagate: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)

    logger.propagate = propagate
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
