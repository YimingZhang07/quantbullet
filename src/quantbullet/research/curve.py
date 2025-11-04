# fmt: off
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from quantbullet.dfutils import get_bins_and_labels
from quantbullet.model.smooth_fit import smooth_monotone_fit, make_monotone_predictor_pchip
from functools import lru_cache
from dataclasses import dataclass
import datetime
from quantbullet.utils.cast import to_date

class MVOCCurve:
    def __init__(self, x_name=None, y_name=None):
        self.x_name = x_name
        self.y_name = y_name

    def build_curve( self, 
                     x_values, 
                     y_values, 
                     dates, 
                     benchmark_date=None,
                     left_bound=None,
                     right_bound=None,
                     step_size=None,  
                     n_grid=100, 
                     alpha=100, 
                     increasing=False,
                     extrapolate='flat' ):
        
        if left_bound is not None:
            self.left_bound = left_bound
        else:
            self.left_bound = min( x_values )
        
        if right_bound is not None:
            self.right_bound = right_bound
        else:
            self.right_bound = max( x_values )

        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = ( self.right_bound - self.left_bound ) / 10

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
        self.smooth_grid_x, self.smooth_fitted_y = smooth_monotone_fit( self.curve_df_['x'], 
                                                                        self.curve_df_['y'], 
                                                                        weights=self.curve_df_['count'], 
                                                                        n_grid=n_grid, 
                                                                        alpha=alpha, 
                                                                        increasing=increasing )
        self.smooth_f_ = make_monotone_predictor_pchip( self.smooth_grid_x, self.smooth_fitted_y, extrapolate=extrapolate )

        return curve_df
    
    def plot_curve( self ):

        x_name = self.x_name or 'X'
        y_name = self.y_name or 'Y'

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
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title(f'{y_name} vs {x_name} Curve')
            ax.grid(True, linestyle="--", alpha=0.5)

        return fig, ax
    
    def predict( self, x_values ):
        return self.smooth_f_( x_values )
    
@dataclass
class MVOCCurveKey:
    Ccy: str
    Rating: str
    AsOfDate: str | pd.Timestamp | datetime.date
    IsDelevered: bool

    def __post_init__( self ):
        # after the dataclass is initialized, convert AsOfDate to date type
        self.AsOfDate = to_date( self.AsOfDate )

class RiskSpreadModel:
    def __init__(self):
        pass
        
    def get_subset_data( self, key: MVOCCurveKey ):
        pass

    @lru_cache(maxsize=None)
    def get_mvoc_curve( self, key: MVOCCurveKey ) -> MVOCCurve:
        df = self.get_subset_data( key )
        curve = MVOCCurve( x_name="MVOC", y_name="RiskSpread" )
        curve.build_curve( x_values=df['MVOC'],
                           y_values=df['RiskSpread'],
                           dates=df['AsOfDate'] )
        return curve

    def predict_single( self, bond ):
        key = MVOCCurveKey(
            Ccy=bond.Ccy,
            Rating=bond.Rating,
            AsOfDate=bond.AsOfDate,
            IsDelevered=bond.IsDelevered
        )
        curve = self.get_mvoc_curve( key )
        return curve.predict( [ bond.MVOC ] )