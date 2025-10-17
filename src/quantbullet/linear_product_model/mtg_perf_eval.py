from dataclasses import dataclass, field
from typing import Union
from quantbullet.plot.utils import scale_scatter_sizes, get_grid_fig_axes, close_unused_axes
from matplotlib import ticker as mticker
from quantbullet.plot.colors import EconomistBrandColor
import pandas as pd

@dataclass
class MtgPerfColnames:
    """Column name mappings for mortgage model performance evaluation.
    The attributes represent the standardized names used in the evaluation code.
    """
    incentive       : str | None = 'incentive'
    cltv            : str | None = 'cltv'
    age             : str | None = 'age'
    response        : str | None = 'actual'
    model_preds     : dict[ str, str ] = field( default_factory=dict )
    dt              : str | None = None
    orig_dt         : str | None = None

    # derived columns
    vintage_year    : str = 'vintage_year'
    vintage_quarter : str = 'vintage_quarter'

    def __post_init__(self):
        for key, val in self.model_preds.items():
            if not hasattr(self, key):
                setattr(self, key, val)

    def std_keys(self) -> list[str]:
        """Standardized keys that should exist in evaluation pipeline."""
        std_keys = [
            k for k in [
                "vintage_year", 
                "vintage_quarter",
                "incentive", 
                "cltv", 
                "age", 
                "response"
            ]
            if getattr(self, k) is not None  # 允许 None 不计入
        ]
        std_keys.extend(self.model_preds.keys())
        return std_keys
    
    def orig_keys(self) -> list[str]:
        """Original column names as provided by the user (ignores None)."""
        orig_keys = [
            v for v in [
                self.vintage_year,
                self.vintage_quarter,
                self.incentive,
                self.cltv,
                self.age,
                self.response,
            ]
            if v is not None
        ]
        orig_keys.extend(self.model_preds.values())
        return orig_keys


class MtgModelPerformanceEvaluator:
    def __init__( self, df: pd.DataFrame, colname_mapping: MtgPerfColnames ):
        self.df = df
        self.colmap = colname_mapping
        self._derive_cols()

    def _derive_cols( self ):
        dt = pd.to_datetime( self.df[ self.colmap.orig_dt ] )
        # Extract vintage year and quarter as categorical variables
        years = dt.dt.year
        year_categories = sorted(years.unique())
        self.df[ self.colmap.vintage_year ] = pd.Categorical(years, categories=year_categories, ordered=True)

        quarters = dt.dt.to_period("Q").astype(str)
        quarter_categories = sorted(quarters.unique(), key=lambda x: (int(x[:4]), int(x[-1])))
        self.df[ self.colmap.vintage_quarter ] = pd.Categorical(quarters, categories=quarter_categories, ordered=True)

    def incentive_by_vintage_year_plots( self, n_quantile_bins: int = 50, n_cols: int = 3, hspace: float = 0.4, wspace: float = 0.3 ):

        X = self.df.copy()

        X['incentive_bins'] = pd.qcut( X[ self.colmap.incentive ], q=n_quantile_bins, duplicates='drop' )

        res = (
            X.groupby( [ self.colmap.vintage_year, 'incentive_bins' ], observed=True )
            .agg(
                actual_mean =( self.colmap.response, 'mean' ),
                count       =( self.colmap.response, 'count' ),
                **{ col: ( orig_col, 'mean' ) for col, orig_col in self.colmap.model_preds.items() }
            )
            .reset_index()
        )

        interval_index = res[ 'incentive_bins' ].cat.categories
        interval_codes = res[ 'incentive_bins' ].cat.codes
        res[ 'bin_right' ] = interval_index.right.take( interval_codes ).to_numpy()

        # Scale sizes for scatter plot
        cmin, cmax = res[ 'count' ].min(), res[ 'count' ].max()
        rescaled_sizes = scale_scatter_sizes( res[ 'count' ], min_size=30, max_size=300, global_min=cmin, global_max=cmax )
        res[ 'size' ] = rescaled_sizes

        vintages = res[ self.colmap.vintage_year ].unique()
        fig, axes = get_grid_fig_axes( n_charts=len( vintages ), n_cols=n_cols )
        fig.subplots_adjust( hspace=hspace, wspace=wspace )

        for ax, vintage in zip( axes, vintages ):
            subdf = res[ res[ self.colmap.vintage_year ] == vintage ]

            for col, orig_col in self.colmap.model_preds.items():
                ax.plot( subdf[ 'bin_right' ], subdf[ col ], label=f'{col} Pred', color=EconomistBrandColor.ECONOMIST_RED )

            sc = ax.scatter(
                subdf[ 'bin_right' ], 
                subdf[ 'actual_mean' ],
                s=subdf[ 'size' ],
                alpha=0.7,
                label='Actual',
                color=EconomistBrandColor.LONDON_70
            )
            ax.set_title(f'Vintage { vintage }')
            ax.yaxis.set_major_formatter( mticker.PercentFormatter( 1.0 ) )
            ax.legend()

        close_unused_axes( axes )
        return fig, axes