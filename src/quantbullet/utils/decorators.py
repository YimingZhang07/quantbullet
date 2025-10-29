import functools
import hashlib
import inspect
import io
import json
import logging
import os
import pickle
import subprocess
import tempfile
import time
import warnings
from collections import OrderedDict
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta
from functools import wraps
from inspect import signature
from threading import RLock
from zoneinfo import ZoneInfo

from .cast import to_date

__all__ = [
    "deprecated",
    "require_fitted",
    "log_runtime",
    "normalize_date_args",
    "external_viewer"
]

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

def disk_cache(cache_dir: str):
    """Cache function outputs to disk using pickle, keying only on kwargs."""
    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            os.makedirs(cache_dir, exist_ok=True)
            # Extract control flags (not part of cache key)
            force_recache = kwargs.pop("force_recache", False)
            expire_days   = kwargs.pop("expire_days", None)

            # Canonicalize: bind args + kwargs → always kwargs
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # Drop self/cls and keep only kwargs
            ba = bound.arguments.copy()
            ba.pop("self", None)
            ba.pop("cls", None)

            # Now everything is in kwargs form
            key_kwargs = dict(sorted(ba.items()))
            key_raw = (func.__qualname__, key_kwargs)

            key_hash = hashlib.sha256(
                json.dumps(key_raw, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]

            base_name = f"{func.__qualname__.replace('.', '_')}_{key_hash}"
            cache_path = os.path.join(cache_dir, base_name + ".pkl")
            meta_path  = os.path.join(cache_dir, base_name + ".json")

            # Load from cache if valid
            if not force_recache and os.path.exists(cache_path):
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                        cache_time = datetime.fromisoformat(meta["timestamp"])
                        if expire_days is not None:
                            if datetime.now(ZoneInfo("America/New_York")) - cache_time > timedelta(days=expire_days):
                                force_recache = True
                if not force_recache:
                    with open(cache_path, "rb") as f:
                        return pickle.load(f)

            # Compute fresh result
            result = func(*args, **kwargs)

            # Save result
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

            # Save metadata (nice readable kwargs only)
            meta = {
                "function": func.__name__,
                "kwargs": {k: repr(v) for k, v in key_kwargs.items()},
                "timestamp": datetime.now(ZoneInfo("America/New_York")).isoformat()
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            return result

        return wrapper
    return decorator

def memory_cache(maxsize: int = 128):
    """
    Decorator to cache function outputs in memory with LRU eviction.
    Keys are built in the same canonical way as disk_cache (kwargs only).
    """
    def decorator(func):
        sig = inspect.signature(func)
        cache = OrderedDict()
        lock = RLock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal cache

            # Canonicalize args+kwargs to kwargs-only
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            ba = bound.arguments.copy()
            ba.pop("self", None)
            ba.pop("cls", None)

            # Build stable key
            key_raw = (func.__qualname__, tuple(sorted(ba.items())))
            key_hash = hashlib.sha256(
                json.dumps(key_raw, sort_keys=True, default=str).encode()
            ).hexdigest()

            # Lookup
            with lock:
                if key_hash in cache:
                    # Move to end (recently used)
                    cache.move_to_end(key_hash)
                    return cache[key_hash]

            # Miss → compute
            result = func(*args, **kwargs)

            with lock:
                cache[key_hash] = result
                cache.move_to_end(key_hash)
                if len(cache) > maxsize:
                    cache.popitem(last=False)  # remove least recently used

            return result

        # Optional: expose cache dict for inspection/clearing
        wrapper._memory_cache = cache
        return wrapper

    return decorator