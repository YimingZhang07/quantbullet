import functools
import warnings
import time
import logging
import io
import tempfile
import subprocess
from contextlib import redirect_stdout
from inspect import signature
from datetime import datetime, date
from functools import wraps

import pandas as pd

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

def run_with_external_viewer(func, *args, open_with="notepad", **kwargs):
    """
    Runs a function and captures its stdout output, saving it to a temporary file.
    Opens the file with the specified external viewer application.
    Parameters
    ----------
        func (Callable): The function to run.
        *args: Positional arguments to pass to the function.
        open_with (str): The external application to open the output file. Options include "notepad", "code", "gedit".
        **kwargs: Keyword arguments to pass to the function.

    Returns
        Any: The result returned by the function.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = func(*args, **kwargs)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(buffer.getvalue())
        file_path = f.name

    if open_with == "notepad":
        subprocess.Popen(["notepad.exe", file_path])
    elif open_with == "code":
        subprocess.Popen(["code", file_path])
    elif open_with == "gedit":
        subprocess.Popen(["gedit", file_path])
    else:
        print(f"Output saved to {file_path}")

    return result

def external_viewer(open_with_arg="open_with", flag_arg="external_view"):
    """
    Decorator to run a function and open its stdout output in an external viewer if a flag argument is set.
    Parameters
    ----------
        open_with (str): The external application to open the output file. Options include "notepad", "code", "gedit".
        flag_arg (str): The name of the boolean keyword argument that triggers the external viewer. Default is "external_view".
    Returns
        function: The decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs.pop(flag_arg, False):
                open_with = kwargs.pop(open_with_arg, "notepad")
                return run_with_external_viewer(func, *args, open_with=open_with, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator