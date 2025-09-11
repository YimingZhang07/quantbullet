import functools
import warnings
import time
import logging
import pandas as pd
from datetime import datetime, date
from inspect import signature
from .cast import to_date

def deprecated(msg: str, is_func_name: bool = False):
    """Decorator to mark a function as deprecated
    
    Parameters
    ----------
    msg : str
        The name of the new function that should be used instead
    is_func_name : bool, optional
        If True, msg is treated as a function name. If False, it is treated as a message. Default is False.

    Returns
    -------
    function
        The decorated function
    """
    def decorator(old_func):
        @functools.wraps(old_func)
        def wrapper(*args, **kwargs):
            if is_func_name:
                warnings.warn(f"Function {old_func.__name__} is deprecated. Use {msg}() instead.", DeprecationWarning)
            else:
                warnings.warn(f"{msg}", DeprecationWarning)
            return old_func(*args, **kwargs)
        return wrapper
    return decorator

def require_fitted(func):
    """Decorator to check if a model has been fitted before calling a method

    The decorated function must be a method of a class with a boolean attribute 'fitted'

    Parameters
    ----------
    func : function
        The function to be decorated

    Returns
    -------
    function
        The decorated function
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")
        return func(self, *args, **kwargs)
    return wrapper

def log_runtime(label: str = None):
    """Decorator to log the runtime of a function or method."""
    def decorator(func):
        logger = logging.getLogger(func.__module__)  # Uses caller's module logger

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            duration = end - start
            log_label = label or func.__qualname__
            logger.info(f"{log_label} completed in {duration:.3f} seconds")
            return result
        return wrapper
    return decorator

def normalize_date_args(*arg_names):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for arg_name in arg_names:
                if arg_name in bound.arguments and bound.arguments[arg_name] is not None:
                    bound.arguments[arg_name] = to_date(bound.arguments[arg_name])

            return func(*bound.args, **bound.kwargs)
        return wrapper
    return decorator
