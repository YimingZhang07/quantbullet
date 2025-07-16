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

def setup_logger(name: str,
                 level: int = logging.INFO,
                 formatter: logging.Formatter = None,
                 force: bool = False, 
                 propagate: bool = False) -> logging.Logger:
    """Set up a logger with the specified name and level.
    
    Parameters
    ----------
    name : str
        The name of the logger.
    level : int
        The logging level for the logger.
    formatter : logging.Formatter, optional
        The formatter to use for the logger's handlers.
    force : bool, optional
        If True, will force the logger to clear existing handlers.
    propagate : bool, optional
        If True, the logger will propagate messages to the parent logger.
        
    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if force:
        logger.handlers.clear()

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        if formatter is None:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                datefmt="%m-%d %H:%M:%S"
            )
        handler.setFormatter(formatter)
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

import logging
import os
from datetime import datetime
import inspect


# below is an example of how to add a file handler to the logger
# def add_file_handler(
#     logger: logging.Logger,
#     file_path: str = None,
#     log_dir: str = "./logs",
#     mode: str = "a",  # 'a' = append, 'w' = overwrite
#     level: int = logging.DEBUG,
#     formatter: logging.Formatter = None,
#     force: bool = False
# ):
#     """
#     Add a file handler to an existing logger.

#     Parameters:
#         logger (Logger): The logger instance.
#         file_path (str): Full file path to log file. If None, a default name is generated.
#         log_dir (str): Directory to store the log file if file_path is not given.
#         mode (str): 'a' to append, 'w' to overwrite.
#         level (int): Logging level for the file handler.
#         formatter (Formatter): Optional formatter. A default will be used if None.
#         force (bool): If True, removes existing FileHandlers before adding.
#     """
#     if formatter is None:
#         formatter = logging.Formatter(
#             "%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
#         )

#     if force:
#         for h in logger.handlers[:]:
#             if isinstance(h, logging.FileHandler):
#                 logger.removeHandler(h)

#     if not file_path:
#         # Generate a default file name
#         frame = inspect.stack()[-1]
#         calling_script = os.path.splitext(os.path.basename(frame.filename))[0]
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         os.makedirs(log_dir, exist_ok=True)
#         file_path = os.path.join(log_dir, f"{calling_script}_{timestamp}.log")

#     file_handler = logging.FileHandler(file_path, mode=mode)
#     file_handler.setLevel(level)
#     file_handler.setFormatter(formatter)

#     logger.addHandler(file_handler)

#     return file_path  # Optional: return path for reference

# I am also curious on whether we can give a list of loggers and write to a single file

# import logging

# class LoggerNameFilter(logging.Filter):
#     def __init__(self, allowed_logger_names):
#         super().__init__()
#         self.allowed = set(allowed_logger_names)

#     def filter(self, record):
#         return record.name in self.allowed

# def add_file_handler_for_loggers(
#     logger_names,
#     file_path,
#     level=logging.DEBUG,
#     mode='a',
#     formatter=None,
# ):
#     if formatter is None:
#         formatter = logging.Formatter(
#             "%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
#         )

#     handler = logging.FileHandler(file_path, mode=mode)
#     handler.setLevel(level)
#     handler.setFormatter(formatter)
#     handler.addFilter(LoggerNameFilter(logger_names))

#     # Attach to the root logger so it can receive propagated messages
#     root_logger = logging.getLogger()
#     root_logger.addHandler(handler)

