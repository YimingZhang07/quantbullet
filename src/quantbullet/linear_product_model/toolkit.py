# Standard library imports
from collections import namedtuple

# Third-party imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.pagesizes import landscape, letter
from sklearn.preprocessing import OneHotEncoder
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from dataclasses import dataclass, field
from collections import defaultdict

# Local application/library imports
from quantbullet.linear_product_model.base import LinearProductModelBase
from quantbullet.plot.utils import get_grid_fig_axes, close_unused_axes, scale_scatter_sizes
from quantbullet.plot.colors import EconomistBrandColor
from quantbullet.preprocessing.transformers import FlatRampTransformer
from quantbullet.dfutils import get_bins_and_labels
from quantbullet.reporting import AdobeSourceFontStyles, PdfChartReport
from quantbullet.reporting.utils import register_fonts_from_package, merge_pdfs
from quantbullet.reporting.formatters import numberarray2string
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
from quantbullet.model.core import FeatureSpec

class LinearProductModelReportMixin:
    @property
    def fit_summary_table_style( self ):
        style = TableStyle([
            ("FONTNAME",      (0,0), (-1,-1), "SourceCodePro-Regular"),  # mono font
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("ALIGN",         (1,0), (1,-1),  "RIGHT"),  # numbers right-aligned
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ])
        return style

    @property
    def feature_config_table_style( self ):
        style = TableStyle([
            ("FONTNAME",      (0,0), (-1,-1), "SourceCodePro-Regular"),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("ALIGN",         (0,0), (0,-1),  "RIGHT"),    # feature names left
            ("ALIGN",         (1,0), (1,-1),  "CENTER"),  # transformer name centered
            ("ALIGN",         (2,0), (2,-1),  "LEFT"),    # params left
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("GRID",          (0,0), (-1,-1), 0.25, colors.lightgrey),
        ])
        return style
    
@dataclass
class ImpliedActualDataCache:
    feature            : str
    agg_df             : pd.DataFrame
    x_grid             : np.ndarray
    x_grid_preds       : np.ndarray

    # derived fields
    bin_right               : np.ndarray = field(init=False)
    bin_implied_actuals     : np.ndarray = field(init=False)
    bin_model_preds         : np.ndarray = field(init=False)
    bin_count               : np.ndarray = field(init=False)

    def __post_init__(self):
        self.bin_right = self.agg_df['feature_bin_right'].values
        self.bin_implied_actuals = self.agg_df['implied_actual_mean'].values
        self.bin_model_preds = self.agg_df['model_pred'].values
        self.bin_count = self.agg_df['count'].values

class LinearProductModelToolkit( LinearProductModelReportMixin ):
    """A toolkit for building, fitting, evaluating, and reporting linear product models."""

    __slots__ = ( 'feature_spec', 'preprocess_config', 'feature_groups_', 'additional_plots', 'subfeatures_',
                  'implied_actual_plot_axes', 'implied_actual_data_caches' )

    """
    feature_groups_ : Dict[str, List[str]]
        A dictionary mapping each feature name to the list of expanded feature names after preprocessing.
        For example, if the original feature is 'age' and it is transformed into 5 spline basis functions,
        then feature_groups_['age'] = ['age_spline_1', 'age_spline_2', 'age_spline_3', 'age_spline_4', 'age_spline_5'].

        For features that do not has any expansion, the list will be the feature name itself, e.g.
        feature_groups_['age'] = ['age'].

    subfeatures_ : List[str]
        A flat list of all expanded feature names across all feature groups.
        This is the concatenation of all lists in feature_groups_.
    """

    def __init__( self, feature_spec: FeatureSpec, preprocess_config: dict = None ):
        self.feature_spec = feature_spec
        self.preprocess_config = preprocess_config
        self.feature_groups_ = defaultdict( list )
        for feature in feature_spec.x:
            self.feature_groups_[ feature ] = []
        self.additional_plots = []

    def fit( self, X ):
        """Fit the preprocessing transformers to the data and prepare for feature expansion."""
        subfeatures = []
        for feature_name in self.feature_groups_.keys():
            if feature_name not in X.columns:
                raise ValueError(f"Feature {feature_name} not found in input DataFrame columns")

            if feature_name in self.preprocess_config:
                transformer = self.preprocess_config[ feature_name ]
                transformer.fit( X[ [ feature_name ] ] )
                subfeatures_for_this_feature = transformer.get_feature_names_out( [ feature_name ] ).tolist()
                self.feature_groups_[ feature_name ] = subfeatures_for_this_feature
                subfeatures.extend( subfeatures_for_this_feature )
            else:
                # if no transformer is given, indicating the feature do not need to be expanded.
                self.feature_groups_[ feature_name ] = [ feature_name ]
                subfeatures.append( feature_name )
        self.subfeatures_ = subfeatures
        return self

    def get_expanded_df( self, X ):
        """Transform the input DataFrame X into the expanded feature DataFrame for model training."""
        dfs = []
        for feature_name in self.feature_groups_.keys():
            if feature_name in self.preprocess_config:
                transformer = self.preprocess_config[ feature_name ]
                if isinstance(transformer, OneHotEncoder):
                    df_transformed = transformer.transform(X[[feature_name]]).toarray()
                else:
                    df_transformed = transformer.transform(X[[feature_name]])
            else:
                df_transformed = X[[feature_name]].to_numpy()
            dfs.append(df_transformed)
        return pd.DataFrame(np.concatenate(dfs, axis=1), columns=self.subfeatures_)

    @property
    def n_feature_groups( self ):
        return len( self.feature_groups_ )
    
    @property
    def categorical_feature_group_names(self):
        return [ feature_name for feature_name in self.feature_groups_.keys() if self.feature_spec[ feature_name].dtype.is_category() ]

    @property
    def numerical_feature_groups_names(self):
        return [ feature_name for feature_name in self.feature_groups_.keys() if self.feature_spec[ feature_name].dtype.is_numeric() ]

    @property
    def feature_groups(self):
        return self.feature_groups_
    
    @property
    def categorical_feature_groups(self):
        return { feature_name: self.feature_groups_[ feature_name ] for feature_name in self.categorical_feature_group_names }
    
    @property
    def numerical_feature_groups(self):
        return { feature_name: self.feature_groups_[ feature_name ] for feature_name in self.numerical_feature_groups_names }
    
    def has_feature_group( self, feature_name ):
        return feature_name in self.feature_groups_.keys()

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

    def sample_data( self, sample_frac, *args, seed=42 ):
        """Randomly subsample multiple arrays/dataframes with the same mask."""
        if not 0 <= sample_frac <= 1:
            raise ValueError("sample_frac must be between 0 and 1")

        lengths = [len(arg) for arg in args]
        if len(set(lengths)) != 1:
            raise ValueError(f"All arguments must have the same length, got {lengths}")

        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        n = lengths[0]
        sample_mask = rng.rand(n) <= sample_frac

        return tuple(arg[sample_mask] for arg in args)
    
    def get_feature_grid_and_predictions( self, feature_series, model, n_points=200 ):
        """Get a grid of feature values and the corresponding model predictions for that feature."""
        x_min, x_max = feature_series.min(), feature_series.max()
        x_grid = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
        y_grid = self.get_single_feature_pred_given_values( feature_series.name, x_grid, model )
        return x_grid, y_grid
    
    def get_single_feature_pred_given_values( self, feature_name, feature_values, model ):
        """Get model predictions for specific feature values."""
        if self.has_feature_group( feature_name ) is False:
            raise ValueError(f"Feature {feature_name} not found in feature_groups_")

        feature_values = np.array( feature_values ).reshape( -1, 1 )
        transformer = self.preprocess_config.get( feature_name, None )
        if transformer is not None:
            transformed_values = transformer.transform( feature_values )
            # make a data container
            single_feature_container = ProductModelDataContainer(
                orig_df = pd.DataFrame( { feature_name : feature_values.ravel() } ),
                expanded_df=pd.DataFrame( transformed_values, columns = model.feature_groups_.get( feature_name ) ),
                feature_groups={ feature_name: model.feature_groups_.get( feature_name ) }
            )

        else:
            # if no transformer is given, indicating the feature do not need to be expanded.
            single_feature_container = ProductModelDataContainer(
                orig_df = pd.DataFrame( { feature_name : feature_values } ),
                expanded_df=pd.DataFrame( feature_values, columns = model.feature_groups_.get( feature_name ) ),
                feature_groups={ feature_name: model.feature_groups_.get( feature_name ) }
            )

        y_values = model.single_feature_group_predict( group_to_include = feature_name, X=single_feature_container, ignore_global_scale=True )
        return y_values

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
        fig, axes = get_grid_fig_axes( n_charts=n_features, n_cols=3 )
        X_sample, y_sample ,train_df_sample = self.sample_data( sample_frac, X, y, train_df )

        for i, feature in enumerate( self.numerical_feature_groups_names ):
            # the data size may be too large to plot all points, so we sample a fraction of the data
            other_blocks_preds = model.leave_out_feature_group_predict(feature, train_df_sample)
            implied_actual = y_sample / other_blocks_preds

            # codes for plotting
            ax = axes[i]
            ax.scatter(X_sample[feature], implied_actual, alpha=0.2, color=EconomistBrandColor.LONDON_70)
            x_grid, this_feature_preds = self.get_feature_grid_and_predictions( X[feature], model )
            ax.plot(x_grid, this_feature_preds, color=EconomistBrandColor.ECONOMIST_RED, label='Model Prediction', linewidth=2)
            ax.set_title(f'{feature} Implied Error', fontdict={'fontsize': 14} )
            ax.set_xlabel(f'{feature} Value', fontdict={'fontsize': 12} )
            ax.set_ylabel('Implied Actual', fontdict={'fontsize': 12} )

        close_unused_axes( axes )
        plt.tight_layout()
        self.implied_actual_plot_axes = axes
        return fig, axes
    
    def get_implied_actual_caches( self, feature_name ):
        return self.implied_actual_data_caches.get( feature_name, None )

    def plot_discretized_implied_errors(
        self, 
        model,
        X: ProductModelDataContainer,
        method: str = 'bin',
        sample_frac: float = 1,
        n_quantile_groups: int | None = 100,
        n_bins: int = 20,
        min_scatter_size: int = 10,
        max_scatter_size: int = 500,
        hspace: float = 0.4,
        wspace: float = 0.3,
    ) -> tuple[object, list[object]]:
        """
        Generate discretized implied errors plots for the model.
        
        Parameters:
        -----------
        method : str
            avg: Calculate implied actual as mean(y/m) for each bin
            bin: Calculate implied actual as sum(y)/sum(m) for each bin
        """
        # Fetch y from container and apply offset if present
        if X.response is None:
            raise ValueError("ProductModelDataContainer.response must be provided for plotting implied errors.")
        y = X.response
        if hasattr(model, 'offset_y') and getattr(model, 'offset_y') is not None:
            y = y + getattr(model, 'offset_y')
        
        # Setup plotting infrastructure
        n_features = len(self.numerical_feature_groups)
        fig, axes = get_grid_fig_axes(n_charts=n_features, n_cols=3)
        fig.subplots_adjust(hspace=hspace, wspace=wspace)
        
        # Sample data via container; preserve alignment across orig/expanded/response
        X_sample = X.sample(sample_frac) if sample_frac < 1 else X
        y_sample = X_sample.response
        
        # Process each feature and create plotting caches
        data_caches = {}
        for feature, subfeatures in self.numerical_feature_groups.items():
            # Create binned dataframe
            binned_df, cutoff_values_right = self._create_bins(X_sample.orig[feature], n_quantile_groups, n_bins)
            
            # Get predictions from other blocks
            other_blocks_preds = model.leave_out_feature_group_predict( feature, X_sample )
            
            # Calculate implied actual based on method
            agg_df = self._calculate_implied_actual_agg(
                binned_df, y_sample, other_blocks_preds, cutoff_values_right, method
            )
            
            # Get feature grid and predictions
            x_grid, x_grid_preds = self.get_feature_grid_and_predictions(
                X_sample.orig[feature], model
            )

            agg_df['model_pred'] = self.get_single_feature_pred_given_values( feature, agg_df['feature_bin_right'], model )
            
            data_caches[feature] = ImpliedActualDataCache(feature, agg_df, x_grid, x_grid_preds)
        
        # Plot all features with consistent sizing
        global_min_count = min(cache.agg_df['count'].min() for _, cache in data_caches.items())
        global_max_count = max(cache.agg_df['count'].max() for _, cache in data_caches.items())
        
        for i, (_, cache) in enumerate(data_caches.items()):
            ax = axes[i]
            sizes = scale_scatter_sizes(
                cache.bin_count,
                min_size=min_scatter_size,
                max_size=max_scatter_size,
                global_min=global_min_count,
                global_max=global_max_count
            )
            
            # Create scatter plot
            ax.scatter(
                cache.bin_right,
                cache.bin_implied_actuals,
                alpha=0.7,
                color=EconomistBrandColor.LONDON_70,
                label='Implied Actual',
                s=sizes
            )
            
            # Plot model prediction line
            ax.plot(
                cache.x_grid, 
                cache.x_grid_preds,
                color=EconomistBrandColor.ECONOMIST_RED,
                label='Model Prediction',
                linewidth=2
            )
            
            ax.set_xlabel(f'{cache.feature}', fontdict={'fontsize': 12})
            ax.set_ylabel('Implied Actual', fontdict={'fontsize': 12})
        
        close_unused_axes(axes)
        self.implied_actual_plot_axes = axes
        self.implied_actual_data_caches = data_caches
        return fig, axes

    def _create_bins(self, feature_series: pd.Series, n_quantile_groups: int | None, n_bins: int) -> tuple[pd.DataFrame, list[float]]:
        """Create binned dataframe based on a feature series.
        
        Parameters
        ----------
        feature_series : pd.Series
            The feature series to bin.
        n_quantile_groups : int, optional
            The number of quantile groups to create.
        n_bins : int
            The number of bins to create if not using quantiles.

        Returns
        -------
        pd.DataFrame
            The binned dataframe with feature bins and cutoff values.
            Columns include:
              - feature: The original feature values.
              - feature_bin: The binned feature values.
              - cutoff_values_right: The right edges of the bins.
        """
        binned_df = pd.DataFrame( { 'feature': feature_series } )
        
        if n_quantile_groups is not None:
            binned_df[ 'feature_bin' ] = pd.qcut( feature_series, n_quantile_groups, duplicates='drop' )
            cutoff_values_right = [ interval.right for interval in binned_df[ 'feature_bin' ].cat.categories ]
        else:
            # Equal-width bins
            cutoff_values = list( np.linspace( feature_series.min(), feature_series.max(), n_bins + 1 ) )
            cutoff_values = cutoff_values[ 1:-1 ]  # Remove first and last to avoid -inf/inf bins
            
            bins, labels = get_bins_and_labels( cutoffs=cutoff_values, decimal_places=4 )
            binned_df['feature_bin'] = pd.cut( binned_df[ 'feature' ], bins=bins, labels=labels )
            cutoff_values_right = cutoff_values + [ feature_series.max() ]
        
        return binned_df, cutoff_values_right

    def _calculate_implied_actual_agg(
        self,
        binned_df: pd.DataFrame,
        y_sample: np.ndarray,
        other_blocks_preds: np.ndarray,
        cutoff_values_right: list[float],
        method: str = 'bin'
    ) -> pd.DataFrame:
        """Calculate aggregated implied actual values based on the specified method."""
        
        if method == 'avg':
            # Simple method: mean(y/m) per bin
            implied_actual = y_sample / other_blocks_preds
            binned_df['implied_actual'] = implied_actual
            
            agg_df = binned_df.groupby('feature_bin', observed=False).agg(
                implied_actual_mean=('implied_actual', 'mean'),
                count=('implied_actual', 'count')
            ).reset_index()
            
        elif method == 'bin':
            # Binned method: sum(y)/sum(m) per bin
            records = []
            for bin_category in binned_df['feature_bin'].cat.categories:
                bin_mask = binned_df['feature_bin'] == bin_category
                sum_y = np.sum(y_sample[bin_mask])
                sum_m = np.sum(other_blocks_preds[bin_mask])
                
                implied_actual = 0 if np.isclose(sum_m, 0) else sum_y / sum_m
                count = np.sum(bin_mask)
                
                records.append((bin_category, implied_actual, count))
            
            agg_df = pd.DataFrame.from_records(
                records, 
                columns=['feature_bin', 'implied_actual_mean', 'count']
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        agg_df['feature_bin_right'] = cutoff_values_right
        return agg_df
    
    def plot_categorical_plots( self, model: LinearProductModelBase, dcontainer: ProductModelDataContainer, sample_frac=1, hspace=0.4, wspace=0.3, method:str='bin' ):
        if hasattr( model, 'offset_y') and getattr( model, 'offset_y') is not None:
            y = dcontainer.response + getattr( model, 'offset_y')

        n_features = len( self.categorical_feature_groups )
        fig, axes = get_grid_fig_axes( n_charts=n_features, n_cols=3 )
        fig.subplots_adjust(hspace=hspace, wspace=wspace)
        dcontainer_sample = dcontainer.sample(sample_frac) if sample_frac < 1 else dcontainer
        X_sample = dcontainer_sample.orig
        y_sample = dcontainer_sample.response

        for i, (feature, subfeatures) in enumerate(self.categorical_feature_groups.items()):
            ax = axes[i]
            other_blocks_preds = model.leave_out_feature_group_predict( feature, dcontainer_sample )
            this_feature_preds = model.single_feature_group_predict( group_to_include=feature, X=dcontainer_sample, ignore_global_scale=True )

            binned_df = pd.DataFrame({
                'feature_bin': X_sample[feature],
                'feature_pred': this_feature_preds
            })

            if method == 'avg':
                implied_actual = y_sample / other_blocks_preds
                binned_df['implied_actual'] = implied_actual
                agg_df = binned_df.groupby( 'feature_bin', observed=False ).agg(
                    implied_actual_mean         = ('implied_actual', 'mean'),
                    this_feature_preds_mean     = ('feature_pred', 'mean'),
                    count                       = ('implied_actual', 'count')
                )
            elif method == 'bin':
                records = []
                for bin_category in binned_df['feature_bin'].cat.categories:
                    bin_mask = binned_df['feature_bin'] == bin_category
                    sum_y = np.sum(y_sample[bin_mask])
                    sum_m = np.sum(other_blocks_preds[bin_mask])

                    implied_actual = 0 if np.isclose(sum_m, 0) else sum_y / sum_m
                    count = np.sum(bin_mask)

                    this_feature_preds_mean = binned_df.loc[ bin_mask, 'feature_pred' ].mean()

                    records.append((bin_category, implied_actual, count, this_feature_preds_mean))

                agg_df = pd.DataFrame.from_records(
                    records,
                    columns=['feature_bin', 'implied_actual_mean', 'count', 'this_feature_preds_mean']
                )
                agg_df.set_index('feature_bin', inplace=True)
            else:
                raise ValueError(f"Unknown method: {method}")

            # plot bar chart for each feature bin
            x = np.arange(len(agg_df.index))  # numeric positions
            width = 0.35  # width of each bar

            ax.bar(x - width/2, agg_df['implied_actual_mean'], width,
                label='Impl Act', alpha=0.7, color = EconomistBrandColor.LONDON_70)
            ax.bar(x + width/2, agg_df['this_feature_preds_mean'], width,
                label='Pred', alpha=0.7, color = EconomistBrandColor.CHICAGO_55)

            # replace categorical x-labels
            ax.set_xticks(x)
            if len( subfeatures ) <= 8:
                ax.set_xticklabels(agg_df.index)
            elif len( subfeatures ) >= 16:
                ax.set_xticklabels(agg_df.index, rotation=90)
            else:
                ax.set_xticklabels(agg_df.index, rotation=45)

            ax.set_xlabel(f'{feature}', fontdict={'fontsize': 12} )
            ax.set_ylabel('Implied Actual', fontdict={'fontsize': 12} )
            ax.legend()

        close_unused_axes( axes )
        self.categorical_plot_axes = axes
        return fig, axes

    def generate_fitting_summary_pdf( self, model, report_name='Model-Report' ):
        pdf_full_path = f"{report_name}-Summary.pdf"

        # register fonts from local package directory
        register_fonts_from_package()

        doc = SimpleDocTemplate(
            pdf_full_path,
            pagesize=landscape(letter),
        )

        story = []
        story.append( Paragraph( f"{report_name} Summary", AdobeSourceFontStyles.SerifItalic ) )

        # ========== Table of Fitting Summary ==========

        fit_summary_data = [
            ["Number of Features",      f"{model.fit_metadata_.n_features:,}"],
            ["Number of Obs",           f"{model.fit_metadata_.n_obs:,}"],
            ["Mean Y",                  f"{model.fit_metadata_.mean_y:,.6f}"],
            ["Model Scalar",            f"{model.global_scalar_:,.6f}"],
            ["Best Loss",               f"{model.best_loss_:,.6f}"],
            ["ftol",                    f"{model.fit_metadata_.ftol}"],
            ["Use QR Decompsition",     f"{model.fit_metadata_.cache_qr_decomp}"],
        ]

        fit_summary_table = Table(fit_summary_data, colWidths=[200, 100])  # adjust widths
        fit_summary_table.setStyle( self.fit_summary_table_style )
        story.append(fit_summary_table)

        story.append( Spacer(2, 20) )

        # ========== Table of Feature Configurations ==========
        feature_config_data = []
        for feature_group_name, transformer in self.feature_config.items():
            if isinstance(transformer, OneHotEncoder):
                feature_config_data.append( [ feature_group_name, "OneHotEncoder", "" ] )
            else:
                text = f"knots={ numberarray2string( transformer.knots ) }"
                feature_config_data.append( [ feature_group_name, 
                                              "FlatRampTransformer", 
                                              Paragraph(text,AdobeSourceFontStyles.MonoCode ) ] )

        feature_config_table = Table( feature_config_data, colWidths=[150, 200, 300] )
        feature_config_table.setStyle( self.feature_config_table_style )
        story.append(feature_config_table)

        doc.build(story)
        return pdf_full_path

    def generate_plots_pdf( self, report_name='Model-Report' ):
        if not hasattr( self, 'implied_actual_axes' ):
            raise ValueError("No error plots found. Please run implied_errors functions first.")

        pdf_full_path = f"{report_name}-Errors.pdf"

        # step 1: add numerical implied errors plots
        chart_pdf = PdfChartReport( pdf_full_path, corner_text=report_name )
        chart_pdf.new_page( layout=( 3,3 ), suptitle = 'Numerical - Implied Errors' )
        chart_pdf.add_external_axes( self.implied_actual_axes )

        # step 2: add categorical error plots
        chart_pdf.new_page( layout=( 2,2 ), suptitle = 'Categorical - Implied Errors' )
        chart_pdf.add_external_axes( self.categorical_plot_axes )

        # Add additional plots
        for additional_axes, layout, suptitle in self.additional_plots:
            chart_pdf.new_page( layout=layout, suptitle=suptitle )
            chart_pdf.add_external_axes( additional_axes )

        chart_pdf.save()
        
        return pdf_full_path
    
    def add_additional_plots( self, axes, layout=(2, 2), suptitle=None ):
        self.additional_plots.append( ( axes, layout, suptitle ) )

    def generate_full_report_pdf( self, model, report_name='Model-Report' ):
        """Generate a full report PDF including fitting summary and error plots."""
        if not hasattr( self, 'implied_actual_axes' ):
            raise ValueError("No error plots found. Please run implied_errors functions first.")

        pdf_full_path = f"{report_name}-Full.pdf"

        pdf1 = self.generate_fitting_summary_pdf( model, report_name=report_name )
        pdf2 = self.generate_plots_pdf( report_name=report_name )
            
        merged_pdf_path = merge_pdfs([pdf1, pdf2], pdf_full_path)

        # cleanup
        os.remove( pdf1 )
        os.remove( pdf2 )
        return merged_pdf_path
    
    def extract_single_feature_group_fit_data( self, X: ProductModelDataContainer, model: LinearProductModelBase, feature_group_name: str ) -> pd.DataFrame:
        """Extract data related to a single feature group for detailed analysis.
        
        Parameters
        ----------
        X : ProductModelDataContainer
            The data container used for model training, containing both original and expanded feature DataFrames.
        model : BaseModel
            The trained model.
        feature_group_name : str
            The name of the feature group to analyze.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the extracted data for the specified feature group.
        """
        s = model.global_scalar_
        m = model.leave_out_feature_group_predict( feature_group_name, X, ignore_global_scale=True )
        subfeatures_df = X.get_container_for_feature_group( feature_group_name ).expanded_df
        X_pred = model.single_feature_group_predict( feature_group_name, X, ignore_global_scale=True )
        result_df = subfeatures_df.copy()
        result_df[ 'y' ] = subfeatures_df.response
        result_df[ 'global_scalar' ] = s
        result_df[ 'other_blocks_pred' ] = m
        result_df[ 'other_blocks_pred_scaled' ] = m * s
        result_df[ 'feature_pred' ] = X_pred
        result_df[ 'full_pred' ] = result_df[ 'other_blocks_pred_scaled' ] * result_df[ 'feature_pred' ]
        result_df[f"{feature_group_name}"] = X[ feature_group_name ].values
        return result_df