from abc import ABC, abstractmethod
from .feature import FeatureSpec

class CloneableModelMixin:
    def _get_clone_args(self):
        """
        Returns the kwargs used to reconstruct the model.
        Subclasses must override this if needed.
        """
        raise NotImplementedError("Subclasses must implement _get_clone_args")

    def new_instance(self):
        """Returns a new model instance with the same constructor args."""
        return self.__class__(**self._get_clone_args())

class Model( ABC, CloneableModelMixin ):
    def __init__( self, feature_spec: FeatureSpec ):
        self.feature_spec = feature_spec
        
    @abstractmethod
    def fit( self, X, y=None ):
        """
        Fit the model to the data.
        
        y is optional, as some models may not require a target variable. We delegate the selection of x and y to the model itself.
        This allows the instance to take all the data at once.
        """
        pass
    
    @abstractmethod
    def predict( self, X ):
        """Predict using the model."""
        pass
    
    def _select_x(self, df):
        return df[self.feature_spec.x]
    
    def _select_y(self, df):
        return df[self.feature_spec.y]
    
    def _get_clone_args(self):
        """Returns the kwargs used to reconstruct the model. Remember to reimplement this in subclasses if more args are needed."""
        return {
            "feature_spec": self.feature_spec,
        }
