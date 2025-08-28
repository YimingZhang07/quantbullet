import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from quantbullet.plot.utils import get_grid_fig_axes, close_unused_axes
from quantbullet.plot.colors import EconomistBrandColor
from sklearn.preprocessing import OneHotEncoder
from quantbullet.preprocessing.transformers import FlatRampTransformer
from quantbullet.dfutils import get_bins_and_labels
from collections import namedtuple

class LinearProductModelToolkit:
    def __init__( self, feature_config = None ):
        self.feature_config = feature_config

    def fit( self, X ):
        col_names = []
        feature_groups = {}
        for colname, transformer in self.feature_config.items():
            transformer.fit( X[[colname]] )
            col_names.extend(transformer.get_feature_names_out())
            feature_groups[colname] = transformer.get_feature_names_out().tolist()
        self.col_names_ = col_names
        self.feature_groups_ = feature_groups
        return self

    def get_train_df( self, X ):
        dfs = []
        for colname, transformer in self.feature_config.items():
            if isinstance(transformer, OneHotEncoder):
                df_transformed = transformer.transform(X[[colname]]).toarray()
            else:
                df_transformed = transformer.fit_transform(X[colname])
            dfs.append(df_transformed)
        return pd.DataFrame(np.concatenate(dfs, axis=1), columns=self.col_names_)
    
    @property
    def n_feature_groups(self):
        return len(self.feature_groups_)
    
    @property
    def categorical_feature_group_names(self):
        return [ name for name, transformer in self.feature_config.items() if isinstance(transformer, OneHotEncoder) ]

    @property
    def numerical_feature_groups_names(self):
        return [ name for name, transformer in self.feature_config.items() if not isinstance(transformer, OneHotEncoder) ]
    
    @property
    def feature_groups(self):
        return self.feature_groups_
    
    @property
    def categorical_feature_groups(self):
        return { name: transformer for name, transformer in self.feature_config.items() if isinstance(transformer, OneHotEncoder) }
    
    @property
    def numerical_feature_groups(self):
        return { name: transformer for name, transformer in self.feature_config.items() if not isinstance(transformer, OneHotEncoder) }

    def plot_model_curves( self, model ):
        n_features = self.n_feature_groups
        _, axes = get_grid_fig_axes( n_charts=n_features, n_cols=3 )

        for i, (feature, transformer) in enumerate(self.feature_config.items()):
            ax = axes[i]  # pick the subplot for this feature

            if isinstance(transformer, OneHotEncoder):
                coef_dict = model.coef_dict[feature]
                categories = []
                values = []
                for subfeature, coef in coef_dict.items():
                    category = subfeature.split('_')[-1]
                    categories.append(category)
                    values.append(coef)
                ax.bar( categories, values, width=0.2, color=EconomistBrandColor.CHICAGO_55 )
                x_label = 'Category'
                
            if isinstance(transformer, FlatRampTransformer):
                # use the knots from the transformer and generate a smooth range for plotting
                knots = transformer.knots
                x_min, x_max = min(knots), max(knots )
                x_grid = np.linspace(x_min, x_max, 200).reshape(-1, 1)
                coef_vector = model.coef_blocks[feature]
                transformed_x_grid = transformer.transform( x_grid )
                y_vals = transformed_x_grid @ coef_vector
                ax.plot( x_grid, y_vals, color=EconomistBrandColor.CHICAGO_45, linewidth=2 )
                x_label = 'Value'

            ax.set_title( feature, fontsize=14 )
            ax.set_xlabel( x_label, fontsize=12 )
            ax.yaxis.set_major_locator( mticker.MaxNLocator( nbins=6 ) )
            for label in ax.get_xticklabels():
                label.set_fontsize(10)
                # label.set_family( 'Source Sans Pro' )
            for label in ax.get_yticklabels():
                label.set_fontsize(10)
                # label.set_family( 'Source Sans Pro' )

        # hide any unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def sample_data( self, X, y, train_df, sample_frac ):
        sample_mask = np.random.rand( len( X ) ) <= sample_frac
        return X[ sample_mask ], y[ sample_mask ], train_df[ sample_mask ]
    
    def get_feature_grid_and_predictions( self, feature_series, transformer, model, n_points=200 ):
        x_min, x_max = feature_series.min(), feature_series.max()
        x_grid = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
        transformed_x_grid = transformer.transform(x_grid)
        coef_vector = model.coef_blocks[ feature_series.name ]
        y_grid = transformed_x_grid @ coef_vector
        return x_grid, y_grid

    def plot_model_implied_errors( self, model, X, y, train_df, sample_frac=0.1 ):
        """For each feature, plot the implied actual vs model prediction

        Implied actual is defined as the ratio of the true target value to the predicted value from the other feature groups.

        Parameters
        ----------
        model : LinearProductModelBase
            The fitted linear product model
        
        X : pd.DataFrame
            The original feature dataframe. there has to be a column 'y' for the target variable.

        train_df : pd.DataFrame
            The transformed feature dataframe used for training the model.

        """
        n_features = len( self.numerical_feature_groups )
        _, axes = get_grid_fig_axes( n_charts=n_features, n_cols=3 )
        X_sample, y_sample ,train_df_sample = self.sample_data( X, y, train_df, sample_frac )

        for i, ( feature, transformer ) in enumerate( self.feature_config.items() ):
            if isinstance(transformer, FlatRampTransformer):
                # the data size may be too large to plot all points, so we sample a fraction of the data
                other_blocks_preds = model.leave_out_feature_group_predict(feature, train_df_sample)
                implied_actual = y_sample / other_blocks_preds

                # codes for plotting
                ax = axes[i]
                ax.scatter(X_sample[feature], implied_actual, alpha=0.2, color=EconomistBrandColor.LONDON_70)
                x_grid, this_feature_preds = self.get_feature_grid_and_predictions( X[feature], transformer, model )
                ax.plot(x_grid, this_feature_preds, color=EconomistBrandColor.ECONOMIST_RED, label='Model Prediction', linewidth=2)
                ax.set_title(f'{feature} Implied Error', fontdict={'fontsize': 14} )
                ax.set_xlabel(f'{feature} Value', fontdict={'fontsize': 12} )
                ax.set_ylabel('Implied Actual', fontdict={'fontsize': 12} )

        close_unused_axes( axes )
        plt.tight_layout()
        plt.show()
        
    def scale_sizes(self, counts, min_size=30, max_size=300, global_min=None, global_max=None):
        return min_size + (max_size - min_size) * (counts - global_min) / (global_max - global_min)

    def plot_discretized_implied_errors( self, model, X, y, train_df, sample_frac=0.1, quantile=None, n_bins=20, min_scatter_size=30, max_scatter_size=300, hspace=0.4, vspace=0.3 ):

        if hasattr( model, 'offset_y') and getattr( model, 'offset_y') is not None:
            y = y + getattr( model, 'offset_y')

        n_features = len( self.numerical_feature_groups )
        fig, axes = get_grid_fig_axes( n_charts=n_features, n_cols=3 )
        fig.subplots_adjust(hspace=hspace, wspace=0.3)
        X_sample, y_sample ,train_df_sample = self.sample_data( X, y, train_df, sample_frac )

        PlottingCache = namedtuple('PlottingCache', ['feature', 'agg_df', 'x_grid', 'this_feature_preds'])
        plotting_caches = []

        for i, ( feature, transformer ) in enumerate( self.numerical_feature_groups.items() ):
            if isinstance(transformer, FlatRampTransformer):
                # bin the feature based on quantiles
                feature_series = X_sample[feature]
                if quantile is not None:
                    # use quantiles to define bins
                    feature_quantiles = feature_series.quantile( np.arange(0.02, 1.01, 0.05) )
                    cutoff_values = feature_quantiles.drop_duplicates().tolist()
                else:
                    # use eual-width bins
                    cutoff_values = list( np.linspace( feature_series.min(), feature_series.max(), n_bins + 1 ) )
                    # remove the first and last value to avoid having bins with -inf and inf
                    cutoff_values = cutoff_values[1:-1]
                bins, labels = get_bins_and_labels( cutoffs=cutoff_values, decimal_places=4 )
                binned_df = pd.DataFrame( { 'feature' : X_sample[feature] } )
                binned_df['feature_bin'] = pd.cut( binned_df['feature'], bins=bins, labels=labels )

                # calculate the implied actual, which is y / prediction from other feature groups
                other_blocks_preds = model.leave_out_feature_group_predict(feature, train_df_sample)
                implied_actual = y_sample / other_blocks_preds
                binned_df['implied_actual'] = implied_actual

                # get the model prediction for this feature
                this_feature_coef = model.coef_blocks[feature]
                binned_df['feature_pred'] = train_df_sample[ self.feature_groups_[feature] ] @ this_feature_coef

                # aggregate by bin
                agg_df = binned_df.groupby('feature_bin', observed=False).agg(
                    implied_actual_mean = ('implied_actual', 'mean'),
                    feature_pred_mean   = ('feature_pred', 'mean'),
                    count               = ('implied_actual', 'count')
                ).reset_index()

                agg_df['feature_bin_right'] = cutoff_values + [X_sample[feature].max()]
                x_grid, this_feature_preds = self.get_feature_grid_and_predictions( X[feature], transformer, model )
                plotting_caches.append( PlottingCache(feature, agg_df, x_grid, this_feature_preds) )

        # get global min and max counts for scaling point sizes
        global_min_count = min( cache.agg_df['count'].min() for cache in plotting_caches )
        global_max_count = max( cache.agg_df['count'].max() for cache in plotting_caches )
        for i, cache in enumerate(plotting_caches):
            feature = cache.feature
            agg_df = cache.agg_df
            x_grid = cache.x_grid
            this_feature_preds = cache.this_feature_preds
            ax = axes[i]
            sizes = self.scale_sizes( agg_df['count'], min_size=min_scatter_size, max_size=max_scatter_size, global_min=global_min_count, global_max=global_max_count )
            ax.scatter( agg_df['feature_bin_right'], 
                        agg_df['implied_actual_mean'], 
                        alpha=0.7, 
                        color=EconomistBrandColor.LONDON_70, 
                        label='Implied Actual',
                        s=sizes )
            ax.plot( x_grid, this_feature_preds, color=EconomistBrandColor.ECONOMIST_RED, label='Model Prediction', linewidth=2 )
            # off the chart title, duplicate with xlabel
            # ax.set_title(f'{feature} Discretized Implied Error', fontdict={'fontsize': 14} )
            ax.set_xlabel(f'{feature}', fontdict={'fontsize': 12} )
            ax.set_ylabel('Implied Actual', fontdict={'fontsize': 12} )
        close_unused_axes( axes )
        # we cannot use tight_layout here because we have adjusted hspace; or else this will override hspace
        # plt.tight_layout()
        plt.show()