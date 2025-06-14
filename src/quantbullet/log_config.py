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


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and return a logger that outputs to the console.

    Parameters
    ----------
    name : str
        Name of the logger.
    
    level : int
        Logging level (default is logging.INFO).

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # more robust than hasHandlers() in some cases
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

        # Avoid log propagation to root (prevents duplicate logs)
        logger.propagate = False

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
