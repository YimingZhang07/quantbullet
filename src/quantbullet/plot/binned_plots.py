import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quantbullet.plot.utils import get_grid_fig_axes, scale_scatter_sizes
from quantbullet.plot.colors import EconomistBrandColor as EBC

# def plot_facet_scatter(
#     df,
#     facet_colname,
#     x_colname,
#     act_colname,
#     pred_colname,
#     weight_colname = None,
#     bins = None,
#     n_bins = 10,
#     min_size = 20,
#     max_size = 200
# ):
#     """Plot facet scatter plots with actual vs predicted values."""
#     facet_col   = df[ facet_colname ]
#     weight_col  = df[ weight_colname ] if weight_colname else None
#     x_col       = df[ x_colname ]
#     act_col     = df[ act_colname ]
#     pred_col    = df[ pred_colname ]
#     bins        = None

#     if weight_col is None:
#         weight_col = pd.Series( 1, index=df.index )

#     df = pd.DataFrame(
#         {
#             'facet'     : facet_col,
#             'x'         : x_col,
#             'weight'    : weight_col,
#             'act'       : act_col,
#             'pred'      : pred_col
#         }
#     )

#     if bins is None:
#         df['bin'] = pd.qcut( df['x'], q = n_bins, duplicates='drop' )
#     else:
#         raise NotImplementedError("Custom bins not implemented yet.")
    
#     df['bin_val'] = df['bin'].apply( lambda x: x.right )

#     agg = (
#         df.groupby( [ "facet", "bin_val" ], observed=False )
#         .apply(
#             lambda g: pd.Series({
#                 "act_mean": np.average( g[ "act" ], weights=g[ "weight" ] ),
#                 "pred_mean": np.average( g[ "pred" ], weights=g[ "weight" ] ),
#                 "count": len( g ),
#                 "weight_mean": g[ "weight" ].mean(),
#                 "weight_sum": g[ "weight" ].sum(),
#             }),
#             include_groups=False
#         )
#         .reset_index()
#     )

#     facets = agg[ 'facet' ].unique()
#     n_charts = len( facets )

#     # Calculate scatter sizes to make sure they are consistent across facets
#     scatter_global_min = agg[ 'count' ].min()
#     scatter_global_max = agg[ 'count' ].max()
#     scatter_sizes = scale_scatter_sizes(
#         agg[ 'count' ],
#         global_min  = scatter_global_min,
#         global_max  = scatter_global_max,
#         min_size    = min_size,
#         max_size    = max_size
#     )

#     agg[ 'scatter_size' ] = scatter_sizes

#     fig, axes = get_grid_fig_axes( n_charts=n_charts, n_cols=3, width=5, height=4 );
#     for i, (ax, c) in enumerate( zip( axes, facets ) ):
#         this_agg = agg[ agg['facet'] == c ]
#         show_legend = ( i == 0 )
#         ax.scatter(
#             this_agg[ 'bin_val' ], this_agg[ 'act_mean' ],
#             color = EBC.LONDON_70, alpha=0.6,
#             s = this_agg[ 'scatter_size' ],
#             label = f'{act_colname}' if show_legend else None
#         )
#         ax.plot(
#             this_agg[ 'bin_val' ], this_agg[ 'pred_mean' ],
#             color = EBC.ECONOMIST_RED, label = f'{pred_colname}' if show_legend else None
#         )
#         ax.set_title( f'{c}' )
#         ax.set_xlabel( x_colname )
#         ax.set_ylabel( act_colname )

#     for size in [ scatter_global_min, (scatter_global_min + scatter_global_max) / 2, scatter_global_max ]:
#         plt.scatter( [], [], 
#                      s = scale_scatter_sizes( pd.Series( [ size ] ), 
#                      min_size=min_size, max_size=max_size, 
#                      global_max=scatter_global_max, global_min=scatter_global_min )[0], 
#                      color=EBC.LONDON_70, label = f'Size: {int(size)}', alpha=0.6 )

#     fig.legend( loc="center right", bbox_to_anchor=(1.0, 0.5), frameon=False, labelspacing=1.5 )

#     return fig, axes



def prepare_binned_data(df, x_col, act_col, pred_col, facet_col=None, weight_col=None, n_bins=10, min_size=20, max_size=200):
    # Setup weights
    weights = df[weight_col] if weight_col else pd.Series(1, index=df.index)
    
    # Create working copy to avoid SettingWithCopy warnings
    tmp = pd.DataFrame({
        'x': df[x_col],
        'act': df[act_col],
        'pred': df[pred_col],
        'weight': weights
    })
    if facet_col:
        tmp['facet'] = df[facet_col]

    # Binning logic
    tmp['bin'] = pd.qcut(tmp['x'], q=n_bins, duplicates='drop')
    tmp['bin_val'] = tmp['bin'].apply(lambda x: x.right)

    # Aggregation
    group_cols = ['facet', 'bin_val'] if facet_col else ['bin_val']
    agg = (
        tmp.groupby(group_cols, observed=False)
        .apply(lambda g: pd.Series({
            "act_mean": np.average(g['act'], weights=g['weight']),
            "pred_mean": np.average(g['pred'], weights=g['weight']),
            "count": len(g)
        }), include_groups=False)
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
    meta = {'global_min': global_min, 'global_max': global_max, 'min_size': min_size, 'max_size': max_size}
    return agg, meta

def draw_act_vs_pred(ax, agg_df, title=None, show_legend=False):
    ax.scatter(
        agg_df['bin_val'], agg_df['act_mean'],
        color=EBC.LONDON_70, alpha=0.6,
        s=agg_df['scatter_size'],
        label='Actual' if show_legend else None
    )
    ax.plot(
        agg_df['bin_val'], agg_df['pred_mean'],
        color=EBC.ECONOMIST_RED,
        label='Predicted' if show_legend else None
    )
    if title:
        ax.set_title(title)
    return ax


def add_size_legend(fig, meta, color, ax_for_handles):
    """Adds the bubble size reference to the LEFT of the figure."""
    test_values = [
        meta['global_min'],
        (meta['global_min'] + meta['global_max']) / 2,
        meta['global_max']
    ]

    scaled_sizes = scale_scatter_sizes(pd.Series(test_values), **meta)

    handles = []
    labels = []
    for val, sz in zip(test_values, scaled_sizes):
        h = ax_for_handles.scatter([], [], s=sz, color=color, alpha=0.6)
        handles.append(h)
        labels.append(f"{int(val)}")

    return handles, labels

def plot_binned_actual_vs_pred(df, x_col, act_col, pred_col, facet_col=None, ax=None, **kwargs):
    # 1. Get data and scaling metadata
    agg, meta = prepare_binned_data(df, x_col, act_col, pred_col, facet_col, **kwargs)
    
    # 2. Setup Layout
    if facet_col is None:
        if ax is None: fig, ax = plt.subplots()
        axes = [ax]
        facets = [None] # Dummy for the loop
    else:
        fig, axes = get_grid_fig_axes(n_charts=len(agg['facet'].unique()))
        facets = agg['facet'].unique()

    # 3. Plotting Loop
    for i, (ax_obj, val) in enumerate(zip(axes, facets)):
        # Filter data if faceting, else use all
        data = agg[agg['facet'] == val] if val else agg
        
        # Use the Worker
        draw_act_vs_pred(
            ax_obj, data, 
            title=str(val) if val else f"{act_col} vs {pred_col}",
            show_legend=(i == 0) # Model legend on first plot
        )

    # 4. Add Global Size Legend
    model_handles, model_labels = axes[0].get_legend_handles_labels()
    size_handles, size_labels = add_size_legend(fig, meta, color=EBC.LONDON_70, ax_for_handles=axes[0])
    handles = model_handles + size_handles
    labels = model_labels + size_labels

    # make room on the right so the legend doesn't overlap/crop
    fig.subplots_adjust(right=0.8)

    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.82, 0.5),
        frameon=False,
        labelspacing=1.5
    )
    
    return fig, axes