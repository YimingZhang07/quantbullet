import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quantbullet.model.feature import DataType
from pygam import LinearGAM, s, f, te
from quantbullet.plot import (
    EconomistBrandColor,
    get_grid_fig_axes
)
from quantbullet.plot.utils import close_unused_axes
from quantbullet.plot.cycles import use_economist_cycle

class WrapperGAM:
    """A wrapper around pygam's LinearGAM to integrate with FeatureSpec and provide additional functionality.
    
    Attributes
    ----------
    feature_term_map_ : dict
        A mapping from feature names to their corresponding pygam terms.

    category_levels_ : dict
        A mapping from categorical feature names to their levels (categories) observed during training.
    """
    __slots__ = [ 'feature_spec', 'gam_', 'formula_', 'feature_term_map_', 'category_levels_' ]

    def __init__( self, feature_spec ):
        self.feature_spec = feature_spec
        self.gam_ = self._build_gam_formula()
        self.category_levels_ = {}

    def _build_gam_formula( self ):
        terms = []
        self.feature_term_map_ = {}
        m = self.feature_spec.all_inputs_order_map
        for i, feature_name in enumerate(self.feature_spec.x):
            feature = self.feature_spec[feature_name]
            
            kwargs = {k: v for k, v in (feature.specs or {}).items() 
                    if v is not None and k in ['spline_order', 'n_splines', 'lam', 'constraints', 'by']}
            
            if feature.dtype == DataType.FLOAT and kwargs.get('by') is not None:
                t = te( i, m[ kwargs['by'] ], **{ k:v for k,v in kwargs.items() if k != 'by' } )
            elif feature.dtype == DataType.FLOAT:
                t = s(i, **kwargs)
            elif feature.dtype == DataType.CATEGORY:
                t = f(i, **kwargs)
            else:
                raise ValueError(f"Unsupported data type: {feature.dtype}")
            
            terms.append(t)
            self.feature_term_map_[feature_name] = t

        if not terms:
            raise ValueError("No terms to combine")
        elif len(terms) == 1:
            self.formula_ = terms[0]
        else:
            self.formula_ = terms[0]
            for term in terms[1:]:
                self.formula_ = self.formula_ + term

        return LinearGAM( self.formula_ )
    
    def fit( self, X, y, weights=None ):
        X_selected = X[ self.feature_spec.all_inputs ].copy()

        # lock in category levels
        for col in self.feature_spec.sec_x_cat + self.feature_spec.x_cat:
            X_selected[col] = X_selected[col].astype('category')
            self.category_levels_[col] = X_selected[col].cat.categories
            X_selected[col] = X_selected[col].cat.codes

        X_selected = np.asarray( X_selected )
        y = np.asarray( y )

        self.gam_.fit( X_selected, y, weights=weights )
        return self
    
    def predict( self, X ):
        X_selected = X[self.feature_spec.all_inputs].copy()

        for col in self.feature_spec.sec_x_cat + self.feature_spec.x_cat:
            # enforce the same categories as training
            X_selected[col] = (
                pd.Categorical(
                    X_selected[col],
                    categories=self.category_levels_[col]
                ).codes
            )

        X_selected = np.asarray( X_selected )
        return self.gam_.predict( X_selected )

    def __repr__(self):
        return f"WrapperGAM( { self.gam_ } )"
    
    def plot_partial_dependence( self, n_cols=3, suptitle=None, scale_y_axis=True ):
        """Plot partial dependence for each feature in the model."""
        # the color cycle of the axes are determined by the plt.rcParams at the time of axes creation
        # therefore we need to set the color cycle before creating the axes
        with use_economist_cycle():
            fig, axes=  get_grid_fig_axes( n_charts= len( self.feature_term_map_ ), n_cols=n_cols )
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        for i, (feature_name, term) in enumerate( self.feature_term_map_.items() ):

            ax = axes.flat[i]

            if term._name == 'spline_term':
                x_grid = self.gam_.generate_X_grid( term=i )
                x_pdep, confi = self.gam_.partial_dependence( term=i, X=x_grid, width=0.95 )
                ax.plot( x_grid[ :, i ], x_pdep, color = EconomistBrandColor.CHICAGO_45 )
                # ax.plot( x_grid[ :, i ], confi, linestyle='--', color='gray' )
                ax.fill_between( x_grid[ :, i ], confi[ :, 0 ], confi[ :, 1 ], alpha=0.2, color = EconomistBrandColor.CHICAGO_45 )
                ax.set_xlabel( f"{ feature_name }", fontdict={ 'fontsize': 12 } )
                ax.set_ylabel( 'Partial Dependence', fontdict={ 'fontsize': 12 } )

            elif term._name == 'tensor_term':
                m = self.feature_spec.all_inputs_order_map
                by = self.feature_spec[ feature_name ].specs.get( 'by' )
                _ = self.gam_.generate_X_grid( term = i )[:, i]
                x_min, x_max = _.min(), _.max()
                x_grid = np.linspace( x_min, x_max, 100 )

                labels = self.category_levels_.get( by )
                codes = list( range( len( labels ) ) )
                
                for code, label in zip( codes, labels ):
                    XX = np.zeros( (100, len( self.feature_spec.all_inputs )) )
                    XX[ :, i ] = x_grid
                    XX[ :, m[ by ] ] = code
                    pdep, confi = self.gam_.partial_dependence( term=i, X=XX, width=0.95 )
                    ax.plot( x_grid, pdep, label=label )
                    ax.fill_between( x_grid, confi[ :, 0 ], confi[ :, 1 ], alpha=0.2 )
                ax.set_xlabel( f"{ feature_name }, by = { by }", fontdict={ 'fontsize': 12 } )
                ax.set_ylabel( 'Partial Dependence', fontdict={ 'fontsize': 12 } )
                ax.legend( title = by )

            elif term._name == 'factor_term':
                # plot a bar chart for categorical features
                labels = self.category_levels_.get( feature_name )
                codes = list( range( len( labels ) ) )
                XX = np.zeros( (len( codes ), len( self.feature_spec.all_inputs )) )
                XX[ :, i ] = codes
                pdep, confi = self.gam_.partial_dependence( term=i, X=XX, width=0.95 )
                # ax.bar( labels, pdep, yerr=[ pdep - confi[:,0], confi[:,1] - pdep ], capsize=5, color = EconomistBrandColor.CHICAGO_45, alpha=0.7 )
                ax.errorbar(labels, pdep,
                            yerr=[pdep - confi[:,0], confi[:,1] - pdep],
                            fmt='o', capsize=5,
                            color=EconomistBrandColor.CHICAGO_45)
                ax.axhline(0, color='gray', linestyle='--', linewidth=1)  # optional baseline
                ax.set_xlabel( f"{ feature_name }", fontdict={ 'fontsize': 12 } )
                ax.set_ylabel( 'Partial Dependence', fontdict={ 'fontsize': 12 } )
            else:
                pass

        if scale_y_axis:
        # Only adjust y-axis for continuous features (spline_term and tensor_term)
            continuous_axes = []
            for i, (feature_name, term) in enumerate(self.feature_term_map_.items()):
                if term._name in ['spline_term', 'tensor_term']:
                    continuous_axes.append(axes.flat[i])
            
            if continuous_axes:
                y_mins = [ax.get_ylim()[0] for ax in continuous_axes]
                y_maxs = [ax.get_ylim()[1] for ax in continuous_axes]
                global_y_min = min(y_mins)
                global_y_max = max(y_maxs)
                
                for ax in continuous_axes:
                    ax.set_ylim(global_y_min, global_y_max)

        if suptitle:
            plt.suptitle( suptitle, fontsize=14 )

        close_unused_axes( axes )
        # plt.tight_layout()
        return fig, axes

    def __getattr__(self, name):
        """Delegate attribute/method access to the underlying GAM model"""
        if hasattr(self.gam_, name):
            attr = getattr(self.gam_, name)
            # If it's a method, we might want to return it directly
            # or wrap it if needed
            return attr
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, key):
        """Delegate indexing to the underlying GAM model if needed"""
        return self.gam_[key]