import functools
import warnings

def deprecated(new_func_name: str):
    """Decorator to mark a function as deprecated
    
    Parameters
    ----------
    new_func : str
        The new function name to use

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