import functools
import warnings
import time
import logging

def deprecated(new_func_name: str):
    """Decorator to mark a function as deprecated
    
    Parameters
    ----------
    new_func_name : str
        The name of the new function that should be used instead

    Returns
    -------
    function
        The decorated function
    """
    def decorator(old_func):
        @functools.wraps(old_func)
        def wrapper(*args, **kwargs):
            warnings.warn(f"Function {old_func.__name__} is deprecated. Use {new_func_name} instead.", DeprecationWarning)
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