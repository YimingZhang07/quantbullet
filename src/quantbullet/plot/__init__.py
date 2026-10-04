"""Public plotting exports; unrelated plot systems load only when requested."""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "colors": ["EconomistBrandColor"],
        "utils": ["get_grid_fig_axes", "close_unused_axes"],
        "cycles": ["use_economist_cycle"],
        "theme": ["PlotTheme", "MINIMAL_THEME"],
        "formatter": ["PlotFormatter"],
        "binned_plots": ["plot_binned_actual_vs_pred"],
        "scatter_binned": ["plot_scatter_multi_y"],
        "dist": ["plot_distributions"],
        "grouped_data": ["BinSpec", "GroupedMeansData", "summarize_grouped_means"],
        "grouped_means": ["GroupedMeansPlot", "GroupedMeansStyle", "DEFAULT_GROUPED_MEANS_STYLE",
                          "draw_grouped_means", "plot_grouped_means"],
    },
)
