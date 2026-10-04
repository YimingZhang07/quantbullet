"""Load utilities on demand so statistics do not import plotting dependencies."""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=["grouped_stats"],
    submod_attrs={
        "plot": ["plot_price_logret_volatility", "plot_price_with_signal", "plot_shared_x"],
        "helper": ["compute_log_returns", "compute_max_drawdown", "compute_sharpe_ratio", "print_metrics"],
        "backtest": ["SimpleDataProvider", "SimpleSignalProvider", "SimplePosition", "SimpleBacktest", "BacktestingCalendar"],
        "stats": ["cross_correlation"],
        "debug": ["debug_cache", "cache_variables", "load_cache_variables", "object_to_pickle",
                  "pickle_to_object", "profile_function", "profile_instance_method"],
        "grouped_stats": ["grouped_weighted_summary"],
    },
)
