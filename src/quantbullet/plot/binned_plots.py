import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from quantbullet.plot.utils import get_grid_fig_axes, scale_scatter_sizes, pretty_int_breaks, close_unused_axes
from quantbullet.plot.colors import EconomistBrandColor as EBC
from quantbullet.plot.cycles import ECONOMIST_LINE_COLORS


def prepare_binned_data(df, x_col, act_col, pred_cols, facet_col=None, weight_col=None, bins=None, n_bins=10, min_size=20, max_size=100):
    """
    Parameters
    ----------
    bins : None, False, 'discrete', or array-like
        - None (default): Use quantile binning with n_bins
        - False or 'discrete': Group by exact x values (no binning)
        - array-like: Custom bin edges for pd.cut()
    """
    # Setup weights
    weights = df[weight_col] if weight_col else pd.Series(1, index=df.index)
    pred_cols = list(pred_cols)
    
    # Create working copy to avoid SettingWithCopy warnings
    tmp = pd.DataFrame({
        'x': df[x_col],
        'act': df[act_col],
        'weight': weights
    })
    for pred_col in pred_cols:
        tmp[pred_col] = df[pred_col]
    if facet_col:
        tmp['facet'] = df[facet_col]

    # Binning logic
    if bins is False or bins == 'discrete':
        # Discrete mode: group by exact x values (no binning)
        tmp['bin'] = tmp['x']
        tmp['bin_val'] = tmp['x']
    elif bins is None:
        # Quantile binning (default)
        tmp['bin'] = pd.qcut(tmp['x'], q=n_bins, duplicates='drop')
        tmp['bin_val'] = tmp['bin'].apply(lambda x: x.right)
    else:
        # Custom bins
        tmp['bin'] = pd.cut(tmp['x'], bins=bins, duplicates='drop')
        tmp['bin_val'] = tmp['bin'].apply(lambda x: x.right)

    # Aggregation
    group_cols = ['facet', 'bin_val'] if facet_col else ['bin_val']
    pred_mean_cols = {pred_col: f"pred_mean__{pred_col}" for pred_col in pred_cols}
    agg = (
        tmp.groupby(group_cols, observed=False)
        .apply(
            lambda g: pd.Series({
                "act_mean": np.average(g['act'], weights=g['weight']),
                "count": len(g),
                **{
                    pred_mean_cols[pred_col]: np.average(g[pred_col], weights=g['weight'])
                    for pred_col in pred_cols
                },
            }),
            include_groups=False
        )
        .reset_index()
    )

    # Calculate global scaling factors
    global_min = agg['count'].min()
    global_max = agg['count'].max()
    
    # Apply scaling (using your existing scale_scatter_sizes helper)
    agg['scatter_size'] = scale_scatter_sizes(
        agg['count'], 
        global_min=global_min, 
        global_max=global_max, 
        min_size=min_size, 
        max_size=max_size
    )

    # We return the metadata along with the data
    meta = {
        'global_min': global_min,
        'global_max': global_max,
        'min_size': min_size,
        'max_size': max_size,
        'pred_mean_cols': pred_mean_cols,
    }
    return agg, meta

def draw_act_vs_pred(
    ax, 
    agg_df, 
    pred_mean_cols, 
    title=None, 
    show_legend=False, 
    act_label="Actual", 
    pred_labels=None,
    pred_colors=None,
    y_transform=None,
):
    # Drop rows where act_mean is NaN
    agg_df = agg_df[agg_df["act_mean"].notna()]

    if agg_df.empty:
        return ax
    
    # Use provided colors or default to Economist line colors
    if pred_colors is None:
        pred_colors = ECONOMIST_LINE_COLORS

    # Apply y-transform if provided
    act_values = agg_df['act_mean']
    if y_transform is not None:
        act_values = y_transform(act_values)

    # Scatter for actuals (gray)
    ax.scatter(
        agg_df['bin_val'], 
        act_values,
        color=EBC.LONDON_70, 
        alpha=0.6,
        s=agg_df['scatter_size'],
        label=act_label if show_legend else None,
        zorder=3,
    )
    
    # Prediction lines
    pred_labels = pred_labels or {}
    for idx, (pred_col, pred_mean_col) in enumerate(pred_mean_cols.items()):
        label = pred_labels.get(pred_col, pred_col) if show_legend else None
        color = pred_colors[idx % len(pred_colors)]
        
        pred_values = agg_df[pred_mean_col]
        if y_transform is not None:
            pred_values = y_transform(pred_values)
        
        ax.plot(
            agg_df['bin_val'], 
            pred_values,
            label=label,
            color=color,
            linewidth=2,
            zorder=4,
        )
    
    # Professional styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    if title:
        ax.set_title(title, fontsize=11, fontweight='600', pad=10)
    
    ax.tick_params(labelsize=9)
    
    return ax


def add_size_legend(fig, meta, color, ax_for_handles):
    """Adds the bubble size reference to the LEFT of the figure."""
    # Use "pretty" ggplot-like breaks so legend labels look clean (10/20/50/100/200...)
    # and avoid awkward midpoints like 18.5.

    legend_vals = pretty_int_breaks(meta["global_min"], meta["global_max"], n=3)
    if not legend_vals:
        legend_vals = [
            int(round(meta["global_min"])),
            int(round((meta["global_min"] + meta["global_max"]) / 2)),
            int(round(meta["global_max"])),
        ]
        legend_vals = sorted(set(legend_vals))

    test_values = [float(v) for v in legend_vals]

    size_meta = {
        "global_min": meta["global_min"],
        "global_max": meta["global_max"],
        "min_size": meta["min_size"],
        "max_size": meta["max_size"],
    }
    scaled_sizes = scale_scatter_sizes(pd.Series(test_values), **size_meta)

    handles = []
    labels = []
    for val, sz in zip(test_values, scaled_sizes):
        h = ax_for_handles.scatter([], [], s=sz, color=color, alpha=0.6)
        handles.append(h)
        labels.append(f"{int(round(val))}")

    return handles, labels

def plot_binned_actual_vs_pred(
    df,
    x_col,
    act_col,
    pred_col,
    facet_col=None,
    bins=None,
    figsize=(6, 4),
    n_cols=3,
    compact_cols=True,
    close_unused=True,
    pred_colors=None,
    y_transform=None,
    **kwargs,
):
    """
    Parameters
    ----------
    bins : None, False, 'discrete', or array-like, optional
        Binning strategy for x_col:
        - None (default): Quantile binning using n_bins parameter
        - False or 'discrete': Group by exact x values (no binning).
          Use this for discrete variables like age in months, year, etc.
        - array-like: Custom bin edges for pd.cut()
        Example: bins=False for age=[12,13,14,...] to get one point per age
        Example: bins=[0, 12, 24, 36, 60, 120, 360] for custom age bands
    pred_colors : list of str, optional
        Colors for prediction lines. Defaults to ECONOMIST_LINE_COLORS.
    y_transform : callable, optional
        Function to transform y-values (both actual and predicted) before plotting.
        Useful for unit conversions (e.g., monthly to annualized).
        Example: `lambda x: x * 12` to annualize monthly values.
    """
    # 1. Get data and scaling metadata
    pred_cols = [pred_col] if isinstance(pred_col, str) else list(pred_col)
    agg, meta = prepare_binned_data(df, x_col, act_col, pred_cols, facet_col, bins=bins, **kwargs)
    
    # 2. Setup Layout
    if facet_col is None:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
        facets = [None] # Dummy for the loop
        
        # Single plot is usually narrower, so we reserve more percentage width for the legend
        layout_rect = [0, 0, 0.75, 1]
        legend_x = 0.77
    else:
        # If faceting, figsize tuple is interpreted as (width per subplot, height per subplot)
        width, height = figsize
        facets = agg['facet'].unique()
        n_charts = len(facets)
        # If we can fit all charts in a single row, optionally avoid extra columns.
        if compact_cols:
            n_cols = min(n_cols, n_charts)
        fig, axes = get_grid_fig_axes(
            n_charts=n_charts,
            n_cols=n_cols,
            width=width,
            height=height,
        )
        
        # Facet grid is usually wider, so we reserve less percentage width
        layout_rect = [0, 0, 0.85, 1]
        legend_x = 0.87

    # 3. Plotting Loop
    for i, (ax_obj, val) in enumerate(zip(axes, facets)):
        # Filter data if faceting, else use all
        data = agg[agg['facet'] == val] if val else agg
        
        draw_act_vs_pred(
            ax_obj, data,
            pred_mean_cols=meta['pred_mean_cols'],
            title=str(val) if val else f"{act_col} vs {', '.join(pred_cols)}",
            show_legend=(i == 0), # Model legend on first plot
            act_label=act_col,
            pred_labels={pred: pred for pred in pred_cols},
            pred_colors=pred_colors,
            y_transform=y_transform,
        )
    if close_unused:
        close_unused_axes(axes)

    # 4. Add Global Size Legend
    model_handles, model_labels = axes[0].get_legend_handles_labels()
    size_handles, size_labels = add_size_legend(fig, meta, color=EBC.LONDON_70, ax_for_handles=axes[0])
    
    # We use two separate legends to allow for a proper "Size" title/header
    # that is bold and aligned correctly, which is hard to do in a single legend.

    # Legend 1: Model (Actual vs Predicted)
    # Placed slightly above the vertical center
    fig.legend(
        model_handles, model_labels,
        loc="lower left",
        bbox_to_anchor=(legend_x, 0.53),
        frameon=False,
        labelspacing=1.2,
    )

    # Legend 2: Size (Bubble sizes)
    # Placed slightly below the vertical center, with a Title
    fig.legend(
        size_handles, size_labels,
        loc="upper left",
        bbox_to_anchor=(legend_x, 0.47),
        frameon=False,
        labelspacing=1.2,
        title="Size",
        title_fontsize='large' # Ensure it stands out
    )
    # Set alignment of title to left if needed, but default center often looks okay for size circles. 
    # If strictly left needed: title_loc='left' (requires matplotlib 3.6+)
    
    # make room on the right so the legend doesn't overlap/crop
    # We use tight_layout with a rect to restrict the axes to the left side
    fig.tight_layout(rect=layout_rect)
    
    return fig, axes


def plot_binned_actual_vs_pred_plotnine(
    df,
    x_col,
    act_col,
    pred_col,
    facet_col=None,
    bins=None,
    figsize=(6, 4),
    n_cols=3,
    compact_cols=True,
    pred_colors=None,
    y_transform=None,
    **kwargs,
):
    """
    Plot binned actual vs predicted using plotnine (ggplot-style).

    Parameters
    ----------
    bins : None, False, 'discrete', or array-like, optional
        Binning strategy for x_col:
        - None (default): Quantile binning using n_bins parameter
        - False or 'discrete': Group by exact x values (no binning).
        - array-like: Custom bin edges for pd.cut()
    pred_colors : list of str, optional
        Colors for prediction lines. Defaults to ECONOMIST_LINE_COLORS.
    y_transform : callable, optional
        Function to transform y-values (both actual and predicted) before plotting.
        Example: `lambda x: x * 12` to annualize monthly values.
    """
    try:
        from plotnine import (
            ggplot,
            aes,
            geom_point,
            geom_line,
            facet_wrap,
            scale_color_manual,
            scale_size,
            scale_x_continuous,
            labs,
            theme_bw,
            theme,
            element_text,
            element_rect,
            element_line,
        )
    except ImportError as e:
        raise ImportError("plotnine is required for plot_binned_actual_vs_pred_plotnine()") from e

    try:
        from mizani.breaks import pretty_breaks
    except ImportError:
        pretty_breaks = None

    try:
        from mizani.formatters import label_number
    except ImportError:
        label_number = None

    pred_cols = [pred_col] if isinstance(pred_col, str) else list(pred_col)
    agg, meta = prepare_binned_data(df, x_col, act_col, pred_cols, facet_col, bins=bins, **kwargs)

    if agg.empty:
        return ggplot()

    if y_transform is not None:
        agg = agg.copy()
        agg["act_mean"] = y_transform(agg["act_mean"])
        for pred_mean_col in meta["pred_mean_cols"].values():
            agg[pred_mean_col] = y_transform(agg[pred_mean_col])

    pred_mean_cols = meta["pred_mean_cols"]
    reverse_map = {v: k for k, v in pred_mean_cols.items()}

    id_vars = ["bin_val", "count"]
    if facet_col:
        id_vars.append("facet")

    lines_df = agg.melt(
        id_vars=id_vars,
        value_vars=list(pred_mean_cols.values()),
        var_name="pred_mean_col",
        value_name="pred_mean",
    )
    lines_df["pred_col"] = lines_df["pred_mean_col"].map(reverse_map)
    lines_df = lines_df[lines_df["pred_mean"].notna()].sort_values("bin_val")

    if pred_colors is None:
        pred_colors = ECONOMIST_LINE_COLORS

    n_charts = agg["facet"].nunique() if facet_col else 1
    if compact_cols and facet_col:
        n_cols = min(n_cols, n_charts)
    n_rows = int(np.ceil(n_charts / n_cols)) if facet_col else 1

    p = (
        ggplot()
        + geom_point(
            agg,
            aes(x="bin_val", y="act_mean", size="count"),
            color=EBC.LONDON_70,
            alpha=0.6,
        )
        + geom_line(
            lines_df,
            aes(x="bin_val", y="pred_mean", color="pred_col", group="pred_col"),
            size=1.2,
        )
        + scale_size(range=(1.5, 6), name="Count")
        + scale_color_manual(values=pred_colors, name="Model")
        + labs(
            title=f"{act_col} vs {', '.join(pred_cols)}",
            x=x_col,
            y=act_col,
        )
        + theme_bw(base_size=11)
        + theme(
            axis_text_x=element_text(rotation=0, ha="center"),
            axis_title_x=element_text(size=10, weight="bold"),
            axis_title_y=element_text(size=10, weight="bold"),
            axis_text_y=element_text(size=9),
            panel_grid_major=element_line(color="#e0e0e0", size=0.3),
            panel_grid_minor=element_line(color="#f0f0f0", size=0.2),
            panel_border=element_rect(color="#4d4d4d", fill="none", size=0.7),
            strip_background=element_rect(fill="white", color="#4d4d4d", size=0.4),
            strip_text=element_text(size=9, weight="bold", color="#333333"),
            legend_position="right",
            legend_title=element_text(size=9, weight="bold"),
            legend_text=element_text(size=8),
            plot_title=element_text(size=11, weight="bold", ha="left"),
            plot_background=element_rect(fill="white"),
        )
    )

    if pretty_breaks is not None and label_number is not None:
        p = p + scale_x_continuous(
            breaks=pretty_breaks(n=6),
            labels=label_number(accuracy=0.01, trim=False),
        )
    else:
        x_vals = agg["bin_val"].to_numpy()
        x_vals = x_vals[np.isfinite(x_vals)]
        if x_vals.size:
            x_min = float(x_vals.min())
            x_max = float(x_vals.max())
            if x_min == x_max:
                breaks = [x_min]
            else:
                breaks = np.linspace(x_min, x_max, num=6)
            labels = [f"{b:.2f}" for b in breaks]
            p = p + scale_x_continuous(breaks=breaks, labels=labels)

    if facet_col:
        p = p + facet_wrap("~facet", ncol=n_cols)

    # figure size (in inches)
    if figsize is not None:
        if facet_col:
            width_per, height_per = figsize
            p = p + theme(figure_size=(width_per * n_cols, height_per * n_rows))
        else:
            p = p + theme(figure_size=figsize)

    return p


def plot_binned_actual_vs_pred_overlay_plotnine(
    df,
    x_col,
    act_col,
    pred_col,
    facet_col=None,
    bins=None,
    figsize=(8, 5),
    pred_colors=None,
    y_transform=None,
    **kwargs,
):
    """
    Plot binned actual vs predicted on a single panel (no faceting),
    with points and lines sharing colors by group.
    """
    try:
        from plotnine import (
            ggplot,
            aes,
            geom_point,
            geom_line,
            scale_color_manual,
            scale_size,
            scale_x_continuous,
            scale_linetype_discrete,
            labs,
            theme_bw,
            theme,
            element_text,
            element_rect,
            element_line,
        )
    except ImportError as e:
        raise ImportError("plotnine is required for plot_binned_actual_vs_pred_overlay_plotnine()") from e

    try:
        from mizani.breaks import pretty_breaks
    except ImportError:
        pretty_breaks = None

    try:
        from mizani.formatters import label_number
    except ImportError:
        label_number = None

    pred_cols = [pred_col] if isinstance(pred_col, str) else list(pred_col)

    if facet_col is None:
        df = df.copy()
        facet_col = "_all_group"
        df[facet_col] = "All"

    agg, meta = prepare_binned_data(df, x_col, act_col, pred_cols, facet_col, bins=bins, **kwargs)
    if agg.empty:
        return ggplot()

    if y_transform is not None:
        agg = agg.copy()
        agg["act_mean"] = y_transform(agg["act_mean"])
        for pred_mean_col in meta["pred_mean_cols"].values():
            agg[pred_mean_col] = y_transform(agg[pred_mean_col])

    pred_mean_cols = meta["pred_mean_cols"]
    reverse_map = {v: k for k, v in pred_mean_cols.items()}

    id_vars = ["bin_val", "count", "facet"]
    lines_df = agg.melt(
        id_vars=id_vars,
        value_vars=list(pred_mean_cols.values()),
        var_name="pred_mean_col",
        value_name="pred_mean",
    )
    lines_df["pred_col"] = lines_df["pred_mean_col"].map(reverse_map)
    lines_df = lines_df[lines_df["pred_mean"].notna()].sort_values(["facet", "bin_val"])

    if pred_colors is None:
        pred_colors = ECONOMIST_LINE_COLORS

    facet_levels = list(pd.unique(agg["facet"]))
    color_map = {
        level: pred_colors[i % len(pred_colors)]
        for i, level in enumerate(facet_levels)
    }

    p = (
        ggplot()
        + geom_point(
            agg,
            aes(x="bin_val", y="act_mean", size="count", color="facet"),
            alpha=0.6,
        )
        + geom_line(
            lines_df,
            aes(
                **({
                    "x": "bin_val",
                    "y": "pred_mean",
                    "color": "facet",
                    "linetype": "pred_col",
                    "group": "interaction(facet, pred_col)",
                } if len(pred_cols) > 1 else {
                    "x": "bin_val",
                    "y": "pred_mean",
                    "color": "facet",
                    "group": "facet",
                })
            ),
            size=1.2,
        )
        + scale_size(range=(1.5, 6), name="Count")
        + scale_color_manual(values=color_map, name=facet_col)
        + labs(
            title=f"{act_col} vs {', '.join(pred_cols)}",
            x=x_col,
            y=act_col,
        )
        + theme_bw(base_size=11)
        + theme(
            axis_text_x=element_text(rotation=0, ha="center"),
            axis_title_x=element_text(size=10, weight="bold"),
            axis_title_y=element_text(size=10, weight="bold"),
            axis_text_y=element_text(size=9),
            panel_grid_major=element_line(color="#e0e0e0", size=0.3),
            panel_grid_minor=element_line(color="#f0f0f0", size=0.2),
            panel_border=element_rect(color="#4d4d4d", fill="none", size=0.7),
            legend_position="right",
            legend_title=element_text(size=9, weight="bold"),
            legend_text=element_text(size=8),
            plot_title=element_text(size=11, weight="bold", ha="left"),
            plot_background=element_rect(fill="white"),
        )
    )

    if len(pred_cols) > 1:
        p = p + scale_linetype_discrete(name="Model")

    if pretty_breaks is not None and label_number is not None:
        p = p + scale_x_continuous(
            breaks=pretty_breaks(n=6),
            labels=label_number(accuracy=0.01, trim=False),
        )
    else:
        x_vals = agg["bin_val"].to_numpy()
        x_vals = x_vals[np.isfinite(x_vals)]
        if x_vals.size:
            x_min = float(x_vals.min())
            x_max = float(x_vals.max())
            if x_min == x_max:
                breaks = [x_min]
            else:
                breaks = np.linspace(x_min, x_max, num=6)
            labels = [f"{b:.2f}" for b in breaks]
            p = p + scale_x_continuous(breaks=breaks, labels=labels)

    if figsize is not None:
        p = p + theme(figure_size=figsize)

    return p