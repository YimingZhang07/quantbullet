import importlib
import inspect
import sys

def reload_defining_module(obj):
    """
    Reload the module that *defines* obj (function/class), and return the
    updated attribute with the same name.

    Predictable behavior:
    - Reloads exactly one module (the defining module)
    - Returns the new object; caller must rebind
    """
    mod = inspect.getmodule(obj)
    if mod is None or mod.__name__ not in sys.modules:
        raise ValueError("Cannot determine a reloadable defining module for this object.")
    mod2 = importlib.reload(sys.modules[mod.__name__])
    return getattr(mod2, obj.__name__)
