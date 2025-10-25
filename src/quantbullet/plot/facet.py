import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quantbullet.plot.utils import get_grid_fig_axes, scale_scatter_sizes
from quantbullet.plot.colors import EconomistBrandColor as EBC

def plot_facet_scatter(
    df,
    facet_colname,
    x_colname,
    act_colname,
    pred_colname,
    weight_colname = None,
    bins = None,
    n_bins = 10,
    min_size = 20,
    max_size = 200
):
    """Plot facet scatter plots with actual vs predicted values."""
    facet_col   = df[ facet_colname ]
    weight_col  = df[ weight_colname ] if weight_colname else None
    x_col       = df[ x_colname ]
    act_col     = df[ act_colname ]
    pred_col    = df[ pred_colname ]
    bins        = None

    if weight_col is None:
        weight_col = pd.Series( 1, index=df.index )

    df = pd.DataFrame(
        {
            'facet': facet_col,
            'x': x_col,
            'weight': weight_col,
            'act': act_col,
            'pred': pred_col
        }
    )

    if bins is None:
        df['bin'] = pd.qcut( df['x'], q = n_bins, duplicates='drop' )
    else:
        raise NotImplementedError("Custom bins not implemented yet.")
    
    df['bin_val'] = df['bin'].apply( lambda x: x.right )

    agg = (
        df.groupby( [ "facet", "bin_val" ], observed=False )
        .apply(
            lambda g: pd.Series({
                "act_mean": np.average( g[ "act" ], weights=g[ "weight" ] ),
                "pred_mean": np.average( g[ "pred" ], weights=g[ "weight" ] ),
                "count": len( g ),
                "weight_mean": g[ "weight" ].mean(),
                "weight_sum": g[ "weight" ].sum(),
            }),
            include_groups=False
        )
        .reset_index()
    )

    facets = agg[ 'facet' ].unique()
    n_charts = len( facets )

    # Calculate scatter sizes to make sure they are consistent across facets
    scatter_global_min = agg[ 'count' ].min()
    scatter_global_max = agg[ 'count' ].max()
    scatter_sizes = scale_scatter_sizes(
        agg[ 'count' ],
        global_min  = scatter_global_min,
        global_max  = scatter_global_max,
        min_size    = min_size,
        max_size    = max_size
    )

    agg[ 'scatter_size' ] = scatter_sizes

    fig, axes = get_grid_fig_axes( n_charts=n_charts, n_cols=3, width=5, height=4 );
    for i, (ax, c) in enumerate( zip( axes, facets ) ):
        this_agg = agg[ agg['facet'] == c ]
        show_legend = ( i == 0 )
        ax.scatter(
            this_agg[ 'bin_val' ], this_agg[ 'act_mean' ],
            color = EBC.LONDON_70, alpha=0.6,
            s = this_agg[ 'scatter_size' ],
            label = f'{act_colname}' if show_legend else None
        )
        ax.plot(
            this_agg[ 'bin_val' ], this_agg[ 'pred_mean' ],
            color = EBC.ECONOMIST_RED, label = f'{pred_colname}' if show_legend else None
        )
        ax.set_title( f'{c}' )
        ax.set_xlabel( x_colname )
        ax.set_ylabel( act_colname )

    for size in [ scatter_global_min, (scatter_global_min + scatter_global_max) / 2, scatter_global_max ]:
        plt.scatter( [], [], 
                     s = scale_scatter_sizes( pd.Series( [ size ] ), 
                     min_size=min_size, max_size=max_size, 
                     global_max=scatter_global_max, global_min=scatter_global_min )[0], 
                     color=EBC.LONDON_70, label = f'Size: {int(size)}', alpha=0.6 )

    fig.legend( loc="center right", bbox_to_anchor=(1.0, 0.5), frameon=False, labelspacing=1.5 )

    return fig, axes