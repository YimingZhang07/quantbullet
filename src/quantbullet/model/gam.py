import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quantbullet.model.feature import DataType
from pygam import LinearGAM, s, f
from quantbullet.plot import (
    EconomistBrandColor,
    get_grid_fig_axes
)

class WrapperGAM:
    def __init__( self, feature_spec ):
        self.feature_spec = feature_spec
        self.gam_ = self._build_gam_formula()

    def _build_gam_formula( self ):
        terms = []
        self.feature_term_map_ = {}
        for i, feature_name in enumerate(self.feature_spec.x):
            feature = self.feature_spec[feature_name]
            
            # One-liner to filter None values from specs
            kwargs = {k: v for k, v in (feature.specs or {}).items() 
                    if v is not None and k in ['spline_order', 'n_splines', 'lam']}
            
            if feature.dtype == DataType.FLOAT:
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
        if isinstance( X, pd.DataFrame ):
            X_selected = X[ self.feature_spec.x ]
        else:
            X_selected = X
        
        X_selected = np.asarray( X_selected )
        y = np.asarray( y )

        self.gam_.fit( X_selected, y, weights=weights )
        return self
    
    def predict( self, X ):
        if isinstance( X, pd.DataFrame ):
            X_selected = X[ self.feature_spec.x ]
        else:
            X_selected = X

        X_selected = np.asarray( X_selected )
        return self.gam_.predict( X_selected )

    def __repr__(self):
        return f"WrapperGAM( { self.gam_ } )"
    
    def plot_partial_dependence( self ):
        fig, axes=  get_grid_fig_axes( n_charts= len( self.feature_spec.x ), n_cols=3 )
        for i, feature_name in enumerate( self.feature_spec.x ):
            ax = axes[ i ]
            x_grid = self.gam_.generate_X_grid( term=i )
            x_pdep, confi = self.gam_.partial_dependence( term=i, X=x_grid, width=0.95)
            ax.plot( x_grid[ :, i ], x_pdep, color = EconomistBrandColor.CHICAGO_45 )
            ax.plot( x_grid[ :, i ], confi, linestyle='--', color='gray' )
            ax.set_xlabel( f"{ feature_name }", fontdict={ 'fontsize': 12 } )
            ax.set_ylabel( 'Partial Dependence', fontdict={ 'fontsize': 12 } )
        plt.tight_layout()
        plt.show()

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