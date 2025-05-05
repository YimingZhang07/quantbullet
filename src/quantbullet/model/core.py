from abc import ABC, abstractmethod
from .feature import FeatureSpec

class Model(ABC):
    def __init__( self, spec: FeatureSpec ):
        self.spec = spec
        
    @abstractmethod
    def fit( self, X, y ):
        """Fit the model to the data."""
        pass
    
    @abstractmethod
    def predict( self, X ):
        """Predict using the model."""
        pass
    
    def _select_x(self, df):
        return df[self.spec.x]
    
    def _select_y(self, df):
        return df[self.spec.y]