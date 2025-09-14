import os
import sys
import pickle
import hashlib
import logging
import inspect
import types
from functools import wraps
from line_profiler import LineProfiler
from datetime import datetime
from quantbullet.log_config import setup_logger
from .decorators import external_viewer

logger = setup_logger("debug_cache")

def debug_cache(func, cache_dir="debug_cache", use_cache=True, force_recache=False, disable_cache=False):
    os.makedirs(cache_dir, exist_ok=True)

    def wrapped(*args, **kwargs):
        if disable_cache:
            logger.info(f"[debug_cache] Caching disabled for {func.__name__}")
            return func(*args, **kwargs)

        key = hashlib.md5(pickle.dumps((func.__name__, args, kwargs))).hexdigest()

        # Search for any cached file matching this key
        matching_files = sorted(
            [f for f in os.listdir(cache_dir) if f.startswith(func.__name__) and f.endswith(f"{key}.pkl")],
            reverse=True  # most recent first
        )

        latest_cache_file = os.path.join(cache_dir, matching_files[0]) if matching_files else None

        if use_cache and not force_recache and latest_cache_file and os.path.exists(latest_cache_file):
            logger.info(f"[debug_cache] Loaded from cache: {latest_cache_file}")
            with open(latest_cache_file, 'rb') as f:
                return pickle.load(f)

        # Create a new cache file with timestamp
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        new_path = os.path.join(cache_dir, f"{func.__name__}_{timestamp}_{key}.pkl")
        logger.info(f"[debug_cache] Computing and caching result to: {new_path}")

        result = func(*args, **kwargs)
        with open(new_path, 'wb') as f:
            pickle.dump(result, f)
        return result

    return wrapped

def cache_variables(save_dir, subfolder=None, **kwargs):
    """
    Save variables to a directory as .pkl files. The directory will be created if it does not exist.
    
    Parameters
    ----------
    save_dir : str
        Directory where the variables will be saved. If it does not exist, it will be created.
    subfolder : str, optional
        Subfolder name to create inside the save_dir. If None, a timestamped folder will be created.
    kwargs : dict
        Variables to save. The keys will be used as the names of the .pkl files.
    """
    import os
    import pandas as pd
    import pickle
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = subfolder or timestamp
    full_path = os.path.join(save_dir, folder_name)

    # only allowed to create a new directory inside the save_dir
    if not os.path.exists(save_dir):
        raise ValueError(f"save_dir {save_dir} does not exist. Please create it first.")
        
    os.makedirs(full_path, exist_ok=True)

    for name, obj in kwargs.items():
        file_path = os.path.join(full_path, f"{name}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    return full_path

def load_cache_variables(load_dir, *var_names, assign_to_globals=False):
    """
    Load cached variables from a directory. The directory should contain .pkl files with the same names as the variables.
    
    Parameters
    ----------
    load_dir : str
        Directory where the cached variables are stored.
    var_names : str
        Names of the variables to load. The function will look for .pkl files with these names in the load_dir.
    assign_to_globals : bool, optional
        If True, the loaded variables will be assigned to the global namespace. Default is False.
    """
    import os
    import pickle

    results = {}
    for name in var_names:
        file_path = os.path.join(load_dir, f"{name}.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
            results[name] = obj
            if assign_to_globals:
                globals()[name] = obj
    return results

def object_to_pickle(obj, filepath):
    """Save an object to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def pickle_to_object(filepath):
    """Load an object from a pickle file."""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj


# -------------------------
# Profiling
# -------------------------

def unwrap_func(func):
    """Unwrap decorated functions to get to the original function."""
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func

@external_viewer(open_with_arg="open_with", flag_arg="open_profile")
def profile_function(func, *args, instance=None, open_profile=False, open_with="notepad", **kwargs):
    """
    Profiles the execution of a given function using line-by-line analysis.

    Parameters
    ----------
        func (Callable): The function object to be profiled. Can be a class method, regular function, or decorated function.
        *args: Positional arguments to pass to the function.
        instance (object, optional): The instance to bind if profiling an instance method. Defaults to None.
        **kwargs: Keyword arguments to pass to the function.
    
    Returns
    -------
        Any: The result returned by the profiled function.

    Side Effects
    ------------
        Prints line-by-line profiling statistics to the console.
    """
    lp = LineProfiler()
    raw_func = unwrap_func(func)
    profiled_func = lp(raw_func)

    if instance is not None and not inspect.ismethod(raw_func):
        profiled_func = types.MethodType(profiled_func, instance)

    result = profiled_func(*args, **kwargs)
    lp.print_stats()
    return result

def profile_instance_method(instance, method_name, *args, open_profile=False, open_with="notepad", **kwargs):
    """
    Profiles an instance method of a given object.
    Parameters
    ----------
        instance (object): The object instance containing the method to be profiled.
        method_name (str): The name of the method to be profiled.
        *args: Positional arguments to pass to the method.
        **kwargs: Keyword arguments to pass to the method.
    """
    func = getattr(instance, method_name)
    return profile_function(func, *args, instance=instance, open_profile=open_profile, open_with=open_with, **kwargs)