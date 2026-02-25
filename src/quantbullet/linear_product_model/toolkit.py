# ===== Standard Library Imports =====
import os
from collections import namedtuple, defaultdict
from dataclasses import dataclass, field

# ===== Third-Party Imports =====
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors

# ===== Local Application/Library Imports =====
from quantbullet.linear_product_model.base import LinearProductModelBase
from quantbullet.linear_product_model.datacontainer import ProductModelDataContainer
from quantbullet.linear_product_model._acceleration import vector_product_numexpr_dict_values
from quantbullet.model.core import FeatureSpec
from quantbullet.plot.utils import get_grid_fig_axes, close_unused_axes, scale_scatter_sizes
from quantbullet.plot.colors import EconomistBrandColor
from quantbullet.preprocessing.transformers import FlatRampTransformer
from quantbullet.reporting import AdobeSourceFontStyles, PdfChartReport
from quantbullet.reporting.utils import register_fonts_from_package, merge_pdfs
from quantbullet.reporting.formatters import numberarray2string
from quantbullet.linear_product_model.mortgage_diagnostics import MortgageColnames, MortgageDiagnostics

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
    """Cached aggregated implied-actual data for a single feature block.

    Downstream consumers can access convenient array fields (bin_right,
    bin_implied_actuals, bin_model_preds, bin_count) without knowing the
    underlying DataFrame column names.
    """
    feature            : str
    agg_df             : pd.DataFrame

    bin_right               : np.ndarray = field(init=False)
    bin_implied_actuals     : np.ndarray = field(init=False)
    bin_model_preds         : np.ndarray = field(init=False)
    bin_count               : np.ndarray = field(init=False)

    def __post_init__(self):
        self.bin_right = self.agg_df['bin_val'].values
        self.bin_implied_actuals = self.agg_df['implied_actual'].values
        self.bin_model_preds = self.agg_df['model_pred'].values
        self.bin_count = self.agg_df['count'].values

class LinearProductModelToolkit( LinearProductModelReportMixin ):
    """A toolkit for building, fitting, evaluating, and reporting linear product models."""

    __slots__ = ( 'feature_spec', 'preprocess_config', 'feature_groups_', 'additional_plots', 'subfeatures_',
                  'implied_actual_bin_config',
                  'implied_actual_plot_axes', 'implied_actual_data_caches', 'perf_plots_incentive_axes' )

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
        self.implied_actual_bin_config: dict = {}

    def clone( self, exclude_preprocess_features: list=None ):
        """Create a copy of the toolkit, optionally excluding certain features from the preprocessing config."""
        # remove the excluded features the preprocess_config
        new_preprocess_config = self.preprocess_config.copy()
        if exclude_preprocess_features is not None:
            for feature in exclude_preprocess_features:
                if feature in new_preprocess_config:
                    new_preprocess_config.pop( feature )
        return LinearProductModelToolkit( feature_spec=self.feature_spec, preprocess_config=new_preprocess_config )

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
        """Transform the original DataFrame X into the expanded feature DataFrame for model training."""
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
    def numerical_feature_group_names(self):
        return [ feature_name for feature_name in self.feature_groups_.keys() if self.feature_spec[ feature_name].dtype.is_numeric() ]

    @property
    def feature_groups(self):
        """Public accessor for the feature group mapping (``{name: [expanded_names]}``)."""
        return self.feature_groups_

    @property
    def categorical_feature_groups(self):
        return { feature_name: self.feature_groups_[ feature_name ] for feature_name in self.categorical_feature_group_names }
    
    @property
    def numerical_feature_groups(self):
        return { feature_name: self.feature_groups_[ feature_name ] for feature_name in self.numerical_feature_group_names }
    
    def has_feature_group( self, feature_name ):
        return feature_name in self.feature_groups_.keys()

    def get_feature_grid_and_predictions( self, feature_series, model, n_points=200 ):
        """Get a grid of feature values and the corresponding model predictions for that feature."""
        x_min, x_max = feature_series.min(), feature_series.max()
        x_grid = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
        y_grid = self.get_single_feature_pred_given_values( feature_series.name, x_grid, model )
        return x_grid, y_grid
    
    def get_single_feature_pred_given_values( self, feature_name, feature_values, model ):
        """Get model predictions for specific feature values."""
        if not self.has_feature_group( feature_name ):
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
                orig_df = pd.DataFrame( { feature_name : feature_values.ravel() } ),
                expanded_df=pd.DataFrame( feature_values, columns = model.feature_groups_.get( feature_name ) ),
                feature_groups={ feature_name: model.feature_groups_.get( feature_name ) }
            )

        y_values = model.single_feature_group_predict( group_to_include = feature_name, X=single_feature_container, ignore_global_scale=True )
        return y_values

    def get_implied_actual_caches( self, feature_name ):
        return self.implied_actual_data_caches.get( feature_name, None )

    @staticmethod
    def _bin_feature_values(series: pd.Series, strategy, n_quantile_groups: int = 100) -> pd.Series:
        """Map feature values to bin representative values.

        Parameters
        ----------
        strategy : None, 'discrete', or numeric
            - None : quantile binning with *n_quantile_groups* (default)
            - 'discrete' : group by exact values (e.g. integer age)
            - numeric : round to that unit then group (e.g. 1000 rounds to nearest $1 K)
        """
        if strategy == 'discrete':
            return series.copy()
        if isinstance(strategy, (int, float)):
            return (series / strategy).round() * strategy
        bins = pd.qcut(series, q=n_quantile_groups, duplicates='drop')
        return bins.apply(lambda iv: iv.right).astype(float)

    def compute_implied_actual_data(
        self,
        model,
        dcontainer: ProductModelDataContainer,
        sample_frac: float = 1,
    ) -> dict[str, pd.DataFrame]:
        """Compute per-feature observation-level implied-actual data.

        For each numerical feature block *f*, the implied actual is
        ``y / (global_scalar * prod(other blocks))``, i.e. what this block
        "should" output given the response and all other blocks.

        For interaction features the result is split by category: keys are
        ``"feature|by_var=cat_val"`` and each DataFrame only contains the
        rows belonging to that category.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keyed by feature name (or ``feature|by=cat`` for interactions).
            Each DataFrame has columns:
            ``feature_value, implied_actual, model_pred, weight``.
        """
        from .base import InteractionCoef

        if dcontainer.response is None:
            raise ValueError("ProductModelDataContainer.response must be provided.")

        dc = dcontainer.sample(sample_frac) if sample_frac < 1 else dcontainer
        y = np.asarray(dc.response, dtype=float)
        if hasattr(model, 'offset_y') and model.offset_y is not None:
            y = y + model.offset_y

        block_preds = {
            feat: model.single_feature_group_predict(feat, dc, ignore_global_scale=True)
            for feat in self.feature_groups_.keys()
        }

        result: dict[str, pd.DataFrame] = {}
        for feature in self.numerical_feature_groups:
            other_preds = model.global_scalar_ * vector_product_numexpr_dict_values(
                data=block_preds, exclude=feature,
            )

            coef = model.coef_.get(feature)
            if isinstance(coef, InteractionCoef):
                cat_series = dc.orig[coef.by]
                X_block = dc.get_expanded_array_for_feature_group(feature)
                for cat_val, cat_coef in coef.categories.items():
                    mask = (cat_series == cat_val).values
                    if hasattr(cat_coef, 'predict'):
                        cat_pred = cat_coef.predict(X_block[mask])
                    else:
                        cat_pred = X_block[mask] @ cat_coef
                    key = f"{feature}|{coef.by}={cat_val}"
                    result[key] = pd.DataFrame({
                        'feature_value': dc.orig[feature].values[mask],
                        'implied_actual': y[mask] / other_preds[mask],
                        'model_pred': cat_pred,
                        'weight': other_preds[mask],
                    })
            else:
                result[feature] = pd.DataFrame({
                    'feature_value': dc.orig[feature].values,
                    'implied_actual': y / other_preds,
                    'model_pred': block_preds[feature],
                    'weight': other_preds,
                })
        return result

    def _aggregate_implied_data(
        self,
        raw_data: dict[str, pd.DataFrame],
        bin_config: dict,
        n_quantile_groups: int,
        agg_method: str,
    ) -> dict[str, pd.DataFrame]:
        """Bin and aggregate per-feature implied-actual data.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keyed by feature name.  Each DataFrame has columns:
            ``bin_val, implied_actual, model_pred, count, feature_name``.
        """
        per_feature: dict[str, pd.DataFrame] = {}

        for feature, df in raw_data.items():
            parent_feature = feature.split('|')[0] if '|' in feature else feature
            bin_vals = self._bin_feature_values(
                df['feature_value'],
                bin_config.get(feature, bin_config.get(parent_feature)),
                n_quantile_groups,
            )

            if agg_method == 'weighted':
                y_orig = df['implied_actual'] * df['weight']
                agg = pd.DataFrame({
                    'bin_val': bin_vals,
                    '_y_orig': y_orig,
                    '_weight': df['weight'],
                    'model_pred': df['model_pred'],
                }).groupby('bin_val', observed=True).agg(
                    _sum_y=('_y_orig', 'sum'),
                    _sum_w=('_weight', 'sum'),
                    model_pred=('model_pred', 'mean'),
                    count=('_y_orig', 'count'),
                ).reset_index()
                agg['implied_actual'] = np.where(
                    np.isclose(agg['_sum_w'], 0), 0,
                    agg['_sum_y'] / agg['_sum_w'],
                )
                agg = agg.drop(columns=['_sum_y', '_sum_w'])
            elif agg_method == 'simple':
                agg = pd.DataFrame({
                    'bin_val': bin_vals,
                    'implied_actual': df['implied_actual'],
                    'model_pred': df['model_pred'],
                }).groupby('bin_val', observed=True).agg(
                    implied_actual=('implied_actual', 'mean'),
                    model_pred=('model_pred', 'mean'),
                    count=('implied_actual', 'count'),
                ).reset_index()
            else:
                raise ValueError(f"Unknown agg_method: {agg_method}")

            agg['feature_name'] = feature
            per_feature[feature] = agg

        return per_feature

    def plot_implied_actuals(
        self,
        model,
        dcontainer: ProductModelDataContainer,
        agg_method: str = 'weighted',
        sample_frac: float = 1,
        n_quantile_groups: int = 100,
        bin_config: dict | None = None,
        min_count: int = 0,
        ylim: tuple | dict | None = None,
        show_lowess: bool = False,
        lowess_frac: float = 0.3,
        n_cols: int = 3,
        figsize: tuple = (5, 4),
        min_scatter_size: int = 10,
        max_scatter_size: int = 500,
    ):
        """Plot implied actuals vs model predictions for each numerical feature.

        For each feature block in a multiplicative model, the implied actual
        is ``y / (global_scalar * product_of_other_blocks)`` -- i.e. what this
        block "should" output given the response and all other blocks.

        Parameters
        ----------
        agg_method : str
            ``'weighted'`` (default): sum(y) / sum(m) per bin -- a ratio of
            totals that down-weights noisy individual observations.
            ``'simple'``: mean(y/m) per bin -- unweighted average of
            individual implied-actual ratios.
        bin_config : dict, optional
            Per-feature binning overrides, merged with
            ``self.implied_actual_bin_config``.  Values can be:
            ``None`` (quantile), ``'discrete'``, or a numeric rounding unit.
        n_quantile_groups : int
            Number of quantile bins for features using default binning.
        min_count : int
            Hide bins with fewer than this many observations.
        ylim : tuple or dict, optional
            Y-axis display range.  A tuple ``(ymin, ymax)`` applies to all
            features; a dict ``{'feature': (ymin, ymax)}`` sets per-feature
            limits.  Data is never modified -- only ``ax.set_ylim`` is called.
        show_lowess : bool
            If True, overlay a LOWESS curve (local regression) on each
            subplot as a non-parametric reference.  Useful for spotting
            curvature the piecewise-linear model may be missing.
        lowess_frac : float
            Fraction of data used for each LOWESS local fit (bandwidth).
            Smaller values follow the data more tightly; larger values
            produce a smoother reference.
        """
        raw_data = self.compute_implied_actual_data(model, dcontainer, sample_frac)
        effective_config = {**self.implied_actual_bin_config, **(bin_config or {})}
        per_feature = self._aggregate_implied_data(
            raw_data, effective_config, n_quantile_groups, agg_method,
        )

        data_caches = {}
        for feature, agg_df in per_feature.items():
            data_caches[feature] = ImpliedActualDataCache(feature, agg_df)
        self.implied_actual_data_caches = data_caches

        features = list(per_feature.keys())
        fig, axes = get_grid_fig_axes(n_charts=len(features), n_cols=n_cols,
                                      width=figsize[0], height=figsize[1])

        global_min_count = min(df['count'].min() for df in per_feature.values())
        global_max_count = max(df['count'].max() for df in per_feature.values())

        for i, feature in enumerate(features):
            ax = axes[i]
            agg = per_feature[feature]

            show = agg if min_count <= 0 else agg[agg['count'] >= min_count]

            sizes = scale_scatter_sizes(
                show['count'],
                min_size=min_scatter_size, max_size=max_scatter_size,
                global_min=global_min_count, global_max=global_max_count,
            )
            ax.scatter(
                show['bin_val'], show['implied_actual'],
                s=sizes, alpha=0.7, color=EconomistBrandColor.LONDON_70,
                label='Implied Actual',
            )
            ax.plot(
                show['bin_val'], show['model_pred'],
                color=EconomistBrandColor.ECONOMIST_RED, linewidth=2,
                label='Model Prediction',
            )

            if show_lowess and len(show) >= 3:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                counts = show['count'].values
                reps = np.maximum(1, np.round(counts / counts.max() * 100)).astype(int)
                x_rep = np.repeat(show['bin_val'].values, reps)
                y_rep = np.repeat(show['implied_actual'].values, reps)
                smoothed = lowess(y_rep, x_rep, frac=lowess_frac)
                ax.plot(
                    smoothed[:, 0], smoothed[:, 1],
                    color=EconomistBrandColor.CHICAGO_45, linewidth=2,
                    linestyle='--', label='LOWESS',
                )

            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Implied Actual', fontsize=12)

            if ylim is not None:
                if isinstance(ylim, dict) and feature in ylim:
                    ax.set_ylim(ylim[feature])
                elif isinstance(ylim, tuple):
                    ax.set_ylim(ylim)

        close_unused_axes(axes)
        self.implied_actual_plot_axes = axes
        return fig, axes

    def plot_categorical_plots( self, model: LinearProductModelBase, dcontainer: ProductModelDataContainer, sample_frac=1, hspace=0.4, wspace=0.3, agg_method:str='weighted' ):
        if hasattr( model, 'offset_y') and getattr( model, 'offset_y') is not None:
            y = dcontainer.response + getattr( model, 'offset_y')

        n_features = len( self.categorical_feature_groups )
        fig, axes = get_grid_fig_axes( n_charts=n_features, n_cols=3 )
        fig.subplots_adjust(hspace=hspace, wspace=wspace)
        dcontainer_sample = dcontainer.sample(sample_frac) if sample_frac < 1 else dcontainer
        X_sample = dcontainer_sample
        y_sample = dcontainer_sample.response

        block_preds = { feature: model.single_feature_group_predict( feature, X_sample, ignore_global_scale=True ) for feature in self.feature_groups_.keys() }
        for i, (feature, subfeatures) in enumerate(self.categorical_feature_groups.items()):
            ax = axes[i]
            other_blocks_preds = model.global_scalar_ * vector_product_numexpr_dict_values( data=block_preds, exclude=feature )
            this_feature_preds = block_preds[ feature ]

            binned_df = pd.DataFrame({
                "feature_bin": X_sample.orig[feature],
                "feature_pred": this_feature_preds,
                "y": y_sample,
                "m": other_blocks_preds,
            })

            if agg_method == 'simple':
                implied_actual = y_sample / other_blocks_preds
                binned_df['implied_actual'] = implied_actual
                agg_df = binned_df.groupby( 'feature_bin', observed=False ).agg(
                    implied_actual_mean         = ('implied_actual', 'mean'),
                    this_feature_preds_mean     = ('feature_pred', 'mean'),
                    count                       = ('implied_actual', 'count')
                )
            elif agg_method == 'weighted':
                agg_df = (
                    binned_df.groupby("feature_bin", observed=False)
                    .agg(
                        sum_y=("y", "sum"),
                        sum_m=("m", "sum"),
                        count=("y", "count"),
                        this_feature_preds_mean=("feature_pred", "mean"),
                    )
                    .reset_index()
                    .set_index("feature_bin")
                )
                agg_df["implied_actual_mean"] = np.where(
                    np.isclose(agg_df["sum_m"], 0), 0, agg_df["sum_y"] / agg_df["sum_m"]
                )
                agg_df.drop(columns=["sum_y", "sum_m"], inplace=True)
            else:
                raise ValueError(f"Unknown agg_method: {agg_method}")

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

    def generate_fitting_summary_pdf( self, model: LinearProductModelBase, report_name='Model-Report' ):
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
        for feature_group_name in self.feature_groups_.keys():
            transformer = self.preprocess_config.get( feature_group_name, None )
            if transformer is None:
                submodel = model.submodels_.get( feature_group_name, None )
                model_name = submodel.model_name
                model_repr = submodel.math_repr()
                feature_config_data.append( [ feature_group_name, 
                                              Paragraph(model_name, AdobeSourceFontStyles.MonoCode),
                                              Paragraph(model_repr, AdobeSourceFontStyles.MonoCode) ] )
            elif isinstance(transformer, OneHotEncoder):
                feature_config_data.append( [ feature_group_name, 
                                              Paragraph("OneHotEncoder", AdobeSourceFontStyles.MonoCode),
                                              Paragraph("", AdobeSourceFontStyles.MonoCode) ] )
            elif isinstance(transformer, FlatRampTransformer):
                text = f"knots={ numberarray2string( transformer.knots ) }"
                feature_config_data.append( [ feature_group_name, 
                                              Paragraph("FlatRampTransformer", AdobeSourceFontStyles.MonoCode),
                                              Paragraph(text, AdobeSourceFontStyles.MonoCode ) ] )
            else:
                raise ValueError(f"Unknown transformer type: {type(transformer)}")

        feature_config_table = Table( feature_config_data, colWidths=[150, 200, 300] )
        feature_config_table.setStyle( self.feature_config_table_style )
        story.append(feature_config_table)

        doc.build(story)
        return pdf_full_path

    def generate_plots_pdf( self, report_name='Model-Report' ):
        if not hasattr( self, 'implied_actual_plot_axes' ):
            raise ValueError("No error plots found. Please run implied_errors functions first.")

        pdf_full_path = f"{report_name}-Errors.pdf"

        # step 1: add numerical implied errors plots
        if hasattr( self, 'implied_actual_plot_axes' ):
            chart_pdf = PdfChartReport( pdf_full_path, corner_text=report_name )
            chart_pdf.new_page( layout=( 3, 3 ), suptitle = 'Numerical - Implied Errors' )
            chart_pdf.add_external_axes( self.implied_actual_plot_axes )

        # step 2: add categorical error plots
        if hasattr( self, 'categorical_plot_axes' ):
            chart_pdf.new_page( layout=( 2, 2 ), suptitle = 'Categorical - Implied Errors' )
            chart_pdf.add_external_axes( self.categorical_plot_axes )

        # step 3: add performance diagnostic plots if available
        if hasattr( self, 'perf_plots_incentive_axes' ):
            chart_pdf.new_page( layout=( 3, 3 ), suptitle = 'Performance Diagnostics - Incentive by Vintage Year' )
            chart_pdf.add_external_axes( self.perf_plots_incentive_axes )

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
        if not hasattr( self, 'implied_actual_plot_axes' ):
            raise ValueError("No error plots found. Please run implied_errors functions first.")

        pdf_full_path = f"{report_name}-Full.pdf"

        pdf1 = self.generate_fitting_summary_pdf( model, report_name=report_name )
        pdf2 = self.generate_plots_pdf( report_name=report_name )
            
        merged_pdf_path = merge_pdfs([pdf1, pdf2], pdf_full_path)

        # cleanup
        os.remove( pdf1 )
        os.remove( pdf2 )
        return merged_pdf_path
    
    def plot_performance_diagnostics( self, model: LinearProductModelBase, X: ProductModelDataContainer, colnames: MortgageColnames ):
        X_orig = X.orig.copy()
        X_orig['model_pred'] = model.predict( X )
        diag = MortgageDiagnostics( df=X_orig, colnames=colnames )
        incentive_fig, incentive_axes = diag.incentive_by_vintage_year_plots( n_cols=3, n_bins=50 )
        self.perf_plots_incentive_axes = incentive_axes
        return incentive_fig, incentive_axes
    
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