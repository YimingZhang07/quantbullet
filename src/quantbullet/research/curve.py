# fmt: off
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from quantbullet.dfutils import get_bins_and_labels
from quantbullet.model.smooth_fit import smooth_monotone_fit

class MVOCCurve:
    def __init__(self, left_bound, right_bound, step_size, x_name, y_name):
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.step_size = step_size
        self.x_name = x_name
        self.y_name = y_name

    def build_curve( self, x_values, y_values, dates, benchmark_date = None ):

        cutoffs = np.arange( self.left_bound, self.right_bound + self.step_size, self.step_size )
        bins, labels = get_bins_and_labels( cutoffs=cutoffs )

        df = pd.DataFrame( { 'x_values': np.array( x_values ), 'dates': np.array( dates ), 'y_values': np.array( y_values ) } )
        df['bins'] = pd.cut( df['x_values'], bins=bins )

        curve_points = []
        half_life = 14

        if benchmark_date is None:
            benchmark_date = pd.to_datetime( max( dates ) )
        else:
            benchmark_date = pd.to_datetime( benchmark_date )

        for category in df['bins'].cat.categories:
            subset = df[ df['bins'] == category ].copy()

            # sometimes the grid we have is too fine that there is no data point in that bin
            if subset.empty:
                continue

            subset[ 'days_diff' ] = ( benchmark_date - subset[ 'dates' ] ).dt.days
            subset[ 'weight' ] = np.exp( -np.log( 2 ) * subset[ 'days_diff' ] / half_life )

            if not math.isinf( category.mid ):
                mvoc_ = category.mid
            elif not math.isinf( category.left ):
                mvoc_ = category.left
            else:
                mvoc_ = category.right

            curve_points.append( {
                'x': mvoc_,
                'y': np.average( subset['y_values'], weights=subset['weight'] ),
                # the weighted count
                'count': subset['weight'].sum()
            } )

        curve_df = pd.DataFrame( curve_points )
        self.curve_df_ = curve_df


        # now do a smooth monotone fit
        self.smooth_grid_x, self.smooth_fitted_y = smooth_monotone_fit( self.curve_df_['x'], self.curve_df_['y'], weights=self.curve_df_['count'], n_grid=100, alpha=100, increasing=False )

        return curve_df
    
    def plot_curve( self ):
        with sns.plotting_context("talk"), sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(
                data=self.curve_df_,
                x='x',
                y='y',
                marker='o',
                ax=ax,
                label='Data Points'
            )
            sns.lineplot(
                x=self.smooth_grid_x,
                y=self.smooth_fitted_y,
                color='red',
                label='Smoothed Fit',
                ax=ax
            )
            ax.set_xlabel(self.x_name)
            ax.set_ylabel(self.y_name)
            ax.set_title(f'{self.y_name} vs {self.x_name} Curve')
            ax.grid(True, linestyle="--", alpha=0.5)

        return fig, ax