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
        "panels": ["PanelSet", "panel_grid"],
        "binned_plots": ["plot_binned_actual_vs_pred"],
        "scatter_binned": ["plot_scatter_multi_y"],
        "dist": ["plot_distributions"],
        "grouped_data": ["BinSpec", "GroupedMeansData", "summarize_grouped_means"],
        "grouped_means": ["GroupedMeansPlot", "GroupedMeansStyle", "DEFAULT_GROUPED_MEANS_STYLE",
                          "PRINT_GROUPED_MEANS_STYLE", "draw_grouped_means", "label_outer_panels",
                          "plot_grouped_means"],
    },
)
