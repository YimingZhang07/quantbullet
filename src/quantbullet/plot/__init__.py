"""Public plotting exports; unrelated plot systems load only when requested."""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "colors": ["EconomistBrandColor"],
        "utils": ["get_grid_fig_axes", "close_unused_axes"],
        "cycles": ["use_economist_cycle"],
        "theme": ["PlotTheme", "MINIMAL_THEME", "PRINT_THEME", "PRINT_RC"],
        "formatter": ["PlotFormatter", "StepPercentFormatter", "compact_number"],
        "panels": ["PanelSet", "panel_grid", "label_outer_panels"],
        "binned_plots": ["plot_binned_actual_vs_pred"],
        "scatter": ["plot_scatter_multi_y"],
        "dist": ["plot_distributions"],
        "binned_means": ["BinSpec", "BinnedMeans", "BinnedMeansPlot", "BinnedMeansStyle",
                         "DEFAULT_BINNED_MEANS_STYLE", "PRINT_BINNED_MEANS_STYLE",
                         "summarize_binned_means", "draw_binned_means", "plot_binned_means"],
    },
)
