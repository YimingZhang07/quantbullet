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

# Local application/library imports
from quantbullet.plot.utils import get_grid_fig_axes, close_unused_axes, scale_scatter_sizes
from quantbullet.plot.colors import EconomistBrandColor
from quantbullet.preprocessing.transformers import FlatRampTransformer
from quantbullet.dfutils import get_bins_and_labels
from quantbullet.reporting import AdobeSourceFontStyles, PdfChartReport
from quantbullet.reporting.utils import register_fonts_from_package, merge_pdfs
from quantbullet.reporting.formatters import numberarray2string

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

class LinearProductModelToolkit( LinearProductModelReportMixin ):
    def __init__( self, feature_config = None ):
        self.feature_config = feature_config
        self.additional_plots = []

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
    
    def get_feature_grid_and_predictions( self, feature_series, transformer, model, n_points=200 ):
        """Get a grid of feature values and the corresponding model predictions for that feature."""
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
        fig, axes = get_grid_fig_axes( n_charts=n_features, n_cols=3 )
        X_sample, y_sample ,train_df_sample = self.sample_data( sample_frac, X, y, train_df )

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
        self.errors_plot_axes = axes
        return fig, axes

    def plot_discretized_implied_errors( self, model, X, y, train_df, sample_frac=1, n_quantile_groups=None, n_bins=20, min_scatter_size=10, max_scatter_size=500, hspace=0.4, vspace=0.3, m_floor=0 ):
        """Generate discretized implied errors plots for the model. This is used for a binary target specifically"""
        if hasattr( model, 'offset_y') and getattr( model, 'offset_y') is not None:
            y = y + getattr( model, 'offset_y')

        n_features = len( self.numerical_feature_groups )
        fig, axes = get_grid_fig_axes( n_charts=n_features, n_cols=3 )
        fig.subplots_adjust(hspace=hspace, wspace=0.3)
        X_sample, y_sample ,train_df_sample = self.sample_data( sample_frac, X, y, train_df )

        PlottingDatCache = namedtuple('PlottingCache', ['feature', 'agg_df', 'x_grid', 'this_feature_preds'])
        plotting_data_caches = []

        for i, ( feature, transformer ) in enumerate( self.numerical_feature_groups.items() ):
            if isinstance(transformer, FlatRampTransformer):
                # bin the feature based on quantiles
                feature_series = X_sample[feature]
                binned_df = pd.DataFrame( { 'feature' : X_sample[feature] } )
                if n_quantile_groups is not None:
                    binned_df['feature_bin'] = pd.qcut( feature_series, n_quantile_groups, duplicates='drop' )
                    cutoff_values = [ interval.right for interval in binned_df['feature_bin'].cat.categories ]
                    cutoff_values_right = cutoff_values
                else:
                    # use eual-width bins
                    cutoff_values = list( np.linspace( feature_series.min(), feature_series.max(), n_bins + 1 ) )
                    # remove the first and last value to avoid having bins with -inf and inf
                    cutoff_values = cutoff_values[1:-1]
                    # the decimal places ensures that bin labels are not duplicated due to rounding...
                    bins, labels = get_bins_and_labels( cutoffs=cutoff_values, decimal_places=4 )
                    binned_df['feature_bin'] = pd.cut( binned_df['feature'], bins=bins, labels=labels )
                    cutoff_values_right = cutoff_values + [feature_series.max()]

                # calculate the implied actual, which is y / prediction from other feature groups
                other_blocks_preds = model.leave_out_feature_group_predict(feature, train_df_sample)
                other_blocks_preds = np.maximum( other_blocks_preds, m_floor )  # avoid division by zero
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

                agg_df['feature_bin_right'] = cutoff_values_right
                x_grid, this_feature_preds = self.get_feature_grid_and_predictions( X[feature], transformer, model )
                plotting_data_caches.append( PlottingDatCache(feature, agg_df, x_grid, this_feature_preds) )

        # since we have multiple subplots, we want to have a common scale for the scatter sizes
        global_min_count = min( cache.agg_df['count'].min() for cache in plotting_data_caches )
        global_max_count = max( cache.agg_df['count'].max() for cache in plotting_data_caches )
        for i, cache in enumerate(plotting_data_caches):
            feature = cache.feature
            agg_df  = cache.agg_df
            x_grid  = cache.x_grid
            this_feature_preds = cache.this_feature_preds
            ax = axes[i]
            sizes = scale_scatter_sizes( agg_df['count'], min_size=min_scatter_size, max_size=max_scatter_size, global_min=global_min_count, global_max=global_max_count )
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
        self.errors_plot_axes = axes
        return fig, axes
    
    def plot_categorical_plots( self, model, X, y, train_df, sample_frac=1, hspace=0.4, vspace=0.3, m_floor=0 ):
        if hasattr( model, 'offset_y') and getattr( model, 'offset_y') is not None:
            y = y + getattr( model, 'offset_y')

        n_features = len( self.categorical_feature_groups )
        fig, axes = get_grid_fig_axes( n_charts=n_features, n_cols=3 )
        fig.subplots_adjust(hspace=hspace, wspace=0.3)
        X_sample, y_sample ,train_df_sample = self.sample_data( sample_frac, X, y, train_df )

        for i, (feature, transformer) in enumerate(self.categorical_feature_groups.items()):
            ax = axes[i]
            # get the implied actuals
            other_blocks_preds = model.leave_out_feature_group_predict( feature, train_df_sample )
            other_blocks_preds = np.maximum( other_blocks_preds, m_floor )
            implied_actual = y_sample / other_blocks_preds

            # get the feature predictions
            this_feature_coef = model.coef_blocks[feature]
            this_feature_preds = train_df_sample[ self.feature_groups_[feature] ] @ this_feature_coef

            binned_df = pd.DataFrame({
                'feature_bin': X_sample[feature],
                'implied_actual': implied_actual,
                'feature_pred': this_feature_preds
            })

            # aggregate and plot
            agg_df = binned_df.groupby( 'feature_bin', observed=False ).agg(
                implied_actual_mean         = ('implied_actual', 'mean'),
                this_feature_preds_mean     = ('feature_pred', 'mean'),
                count                       = ('implied_actual', 'count')
            )


            # plot bar chart for each feature bin
            x = np.arange(len(agg_df.index))  # numeric positions
            width = 0.35  # width of each bar

            ax.bar(x - width/2, agg_df['implied_actual_mean'], width,
                label='Impl Act', alpha=0.7, color = EconomistBrandColor.LONDON_70)
            ax.bar(x + width/2, agg_df['this_feature_preds_mean'], width,
                label='Pred', alpha=0.7, color = EconomistBrandColor.CHICAGO_55)

            # replace categorical x-labels
            ax.set_xticks(x)
            if len( transformer.get_feature_names_out() ) <= 8:
                ax.set_xticklabels(agg_df.index)
            elif len( transformer.get_feature_names_out() ) >= 16:
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

    def generate_error_plots_pdf( self, report_name='Model-Report' ):
        if not hasattr( self, 'errors_plot_axes' ):
            raise ValueError("No error plots found. Please run implied_errors functions first.")

        pdf_full_path = f"{report_name}-Errors.pdf"

        chart_pdf = PdfChartReport( pdf_full_path, corner_text=report_name )
        chart_pdf.new_page( layout=( 3,3 ), suptitle = 'Numerical - Implied Errors' )
        chart_pdf.add_external_axes( self.errors_plot_axes )

        chart_pdf.new_page( layout=( 2,2 ), suptitle = 'Categorical - Implied Errors' )
        chart_pdf.add_external_axes( self.categorical_plot_axes )

        # Add additional plots if any
        for additional_axes, layout, suptitle in self.additional_plots:
            chart_pdf.new_page( layout=layout, suptitle=suptitle )
            chart_pdf.add_external_axes( additional_axes )

        chart_pdf.save()

        return pdf_full_path
    
    def add_additional_plots( self, axes, layout=(2, 2), suptitle=None ):
        self.additional_plots.append( ( axes, layout, suptitle ) )

    def generate_full_report_pdf( self, model, report_name='Model-Report' ):
        """Generate a full report PDF including fitting summary and error plots."""
        if not hasattr( self, 'errors_plot_axes' ):
            raise ValueError("No error plots found. Please run implied_errors functions first.")

        pdf_full_path = f"{report_name}-Full.pdf"

        pdf1 = self.generate_fitting_summary_pdf( model, report_name=report_name )
        pdf2 = self.generate_error_plots_pdf( report_name=report_name )
            
        merged_pdf_path = merge_pdfs([pdf1, pdf2], pdf_full_path)

        # cleanup
        os.remove( pdf1 )
        os.remove( pdf2 )
        return merged_pdf_path