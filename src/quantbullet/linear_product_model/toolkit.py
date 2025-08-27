import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from quantbullet.plot.utils import get_grid_fig_axes, close_unused_axes
from quantbullet.plot.colors import EconomistBrandColor
from sklearn.preprocessing import OneHotEncoder
from quantbullet.preprocessing.transformers import FlatRampTransformer

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
    def categorical_feature_groups(self):
        return [ name for name, transformer in self.feature_config.items() if isinstance(transformer, OneHotEncoder) ]

    @property
    def numerical_feature_groups(self):
        return [ name for name, transformer in self.feature_config.items() if not isinstance(transformer, OneHotEncoder) ]

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
        sample_mask = np.random.rand( len( X ) ) < sample_frac
        return X[ sample_mask ], y[ sample_mask ], train_df[ sample_mask ]

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

        for i, ( feature, transformer ) in enumerate( self.feature_config.items() ):
            if isinstance(transformer, FlatRampTransformer):
                # the data size may be too large to plot all points, so we sample a fraction of the data
                X_sample, y_sample ,train_df_sample = self.sample_data( X, y, train_df, sample_frac )
                other_blocks_preds = model.leave_out_feature_group_predict(feature, train_df_sample)
                implied_actual = y_sample / other_blocks_preds

                # codes for plotting
                ax = axes[i]
                ax.scatter(X_sample[feature], implied_actual, alpha=0.2, color=EconomistBrandColor.LONDON_70)
                this_feature_coef = model.coef_blocks[feature]
                x_min, x_max = X[feature].min(), X[feature].max()
                x_grid = np.linspace(x_min, x_max, 200).reshape(-1, 1)
                this_feature_preds = transformer.transform(x_grid) @ this_feature_coef
                ax.plot(x_grid, this_feature_preds, color=EconomistBrandColor.ECONOMIST_RED, label='Model Prediction', linewidth=2)
                ax.set_title(f'{feature} Implied Error', fontdict={'fontsize': 14} )
                ax.set_xlabel(f'{feature} Value', fontdict={'fontsize': 12} )
                ax.set_ylabel('Implied Actual', fontdict={'fontsize': 12} )

        close_unused_axes( axes )
        plt.tight_layout()
        plt.show()