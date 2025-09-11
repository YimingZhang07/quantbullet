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
            
            # One-liner to filter None values from specs
            kwargs = {k: v for k, v in (feature.specs or {}).items() 
                    if v is not None and k in ['spline_order', 'n_splines', 'lam', 'constraints', 'by']}
            
            if feature.dtype == DataType.FLOAT and kwargs.get('by') is not None:
                t = te( i, m[ kwargs['by'] ], **{ k:v for k,v in kwargs.items() if k != 'by' } )
            elif feature.dtype == DataType.FLOAT:
                t = s(i, **kwargs)
            elif feature.dtype == DataType.CATEGORICAL:
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
        for col in self.feature_spec.sec_x_cat:
            X_selected[col] = X_selected[col].astype('category')
            self.category_levels_[col] = X_selected[col].cat.categories
            X_selected[col] = X_selected[col].cat.codes

        X_selected = np.asarray( X_selected )
        y = np.asarray( y )

        self.gam_.fit( X_selected, y, weights=weights )
        return self
    
    def predict( self, X ):
        X_selected = X[self.feature_spec.all_inputs].copy()

        for col in self.feature_spec.sec_x_cat:
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

            else:
                pass

        # all the y axes share the same scale
        # find the global y min/max first and then set the same ylim
        if scale_y_axis:
            y_mins = [ ax.get_ylim()[ 0 ] for ax in axes.flat[:len(self.feature_term_map_)] ]
            y_maxs = [ ax.get_ylim()[ 1 ] for ax in axes.flat[:len(self.feature_term_map_)] ]
            global_y_min = min( y_mins )
            global_y_max = max( y_maxs )
            for ax in axes.flat[ :len( self.feature_term_map_ ) ]:
                ax.set_ylim( global_y_min, global_y_max )

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