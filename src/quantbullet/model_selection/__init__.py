import importlib

_lazy_by_module = {
    ".validation"         : [ "CrossValidationResult", "OptunaStudyResult", "OptunaCVOptimizer", "time_series_cv_predict" ],
    ".split"              : [ "TimeSeriesDailyRollingSplit" ]
}

_lazy_map = {
    name: module for module, names in _lazy_by_module.items() for name in names
}

# This is an attribute level lazy loading mechanism, and therefore we use caching to globals()

def __getattr__(name):
    if name in _lazy_map:
        module_path = _lazy_map[name]
        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, name)
        globals()[name] = val  # cache it
        return val
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
    return list(globals().keys()) + list(_lazy_map.keys())