import keyword
import warnings
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict
from dataclasses import dataclass, field
from quantbullet.core.enums import DataType
    
class FeatureRole(Enum):
    MODEL_INPUT         = "model_input"
    TARGET              = "target"
    AUXILIARY_TARGET    = "auxiliary_target"
    REFERENCE           = "reference"
    GROUPING            = "grouping"
    SECONDARY_INPUT     = "secondary_input"

@dataclass
class Feature:
    name    : str
    dtype   : DataType
    role    : FeatureRole
    specs   : Dict[str, Any] = field(default_factory=dict)

class FeatureSpec:
    def __init__(self, features: list[Feature]):
        self._features = features
        self._classify_features()
        self._build_typed_feature_namespace()
        self._build_lookup_maps()

    def _build_lookup_maps(self):
        """Build direct lookup dictionaries for O(1) access."""
        self._name_to_feature = {f.name: f for f in self._features}
        self._name_to_index = {f.name: i for i, f in enumerate(self._features)}

    @property
    def all_inputs( self ) -> list[str]:
        """Get all model input feature names."""
        return self.x + self.sec_x
    
    @property
    def all_inputs_order_map( self ) -> Dict[str, int]:
        """Get a map of all input feature names to their order index."""
        return { name: idx for idx, name in enumerate( self.all_inputs ) }

    def _classify_features(self):
        """Classify features into model inputs, target, references, etc."""
        self.x = [f.name for f in self._features if f.role == FeatureRole.MODEL_INPUT]
        self.x_num = [f.name for f in self._features if f.role == FeatureRole.MODEL_INPUT and f.dtype.is_numeric()]
        self.x_cat = [f.name for f in self._features if f.role == FeatureRole.MODEL_INPUT and f.dtype.is_categorical()]
        self.sec_x = [f.name for f in self._features if f.role == FeatureRole.SECONDARY_INPUT]
        self.sec_x_num = [f.name for f in self._features if f.role == FeatureRole.SECONDARY_INPUT and f.dtype.is_numeric()]
        self.sec_x_cat = [f.name for f in self._features if f.role == FeatureRole.SECONDARY_INPUT and f.dtype.is_categorical()]
        self.refs = [f.name for f in self._features if f.role == FeatureRole.REFERENCE]

        # Expect exactly one target
        targets = [f.name for f in self._features if f.role == FeatureRole.TARGET]
        if len(targets) != 1:
            raise ValueError(f"Expected exactly one target feature, got: {targets}")
        self.y = targets[0]
        
    def _build_typed_feature_namespace(self):
        """Build feature namespace with type annotations."""
        # Initialize class annotations if they don't exist
        if not hasattr(self.__class__, '__annotations__'):
            self.__class__.__annotations__ = {}
        
        # Build valid attribute names and add type annotations
        self._valid_feature_attrs = set()
        
        for f in self._features:
            attr_name = f.name
            if attr_name.isidentifier() and not keyword.iskeyword(attr_name):
                if not hasattr(self, attr_name):
                    self._valid_feature_attrs.add(attr_name)
                    # Add type annotation to class
                    self.__class__.__annotations__[attr_name] = Feature
                    # Set actual attribute
                    setattr(self, attr_name, f)

    # Direct O(1) access methods
    def get_feature_by_name(self, name: str) -> Feature | None:
        """Get feature object by name - O(1) lookup."""
        return self._name_to_feature.get(name)
    
    def get_feature_by_index(self, index: int) -> Feature | None:
        """Get feature object by index - O(1) lookup."""
        return self._features[index] if 0 <= index < len(self._features) else None
    
    def get_feature_index(self, name: str) -> int:
        """Get index of feature by name - O(1) lookup."""
        return self._name_to_index.get(name, -1)
    
    def has_feature(self, name: str) -> bool:
        """Check if feature exists - O(1) lookup."""
        return name in self._name_to_feature
    
    def __getitem__(self, key):
        """Support both string (name) and integer (index) access."""
        if isinstance(key, str):
            feature = self._name_to_feature.get(key)
            if feature is None:
                raise KeyError(f"Feature '{key}' not found")
            return feature
        elif isinstance(key, int):
            if not (0 <= key < len(self._features)):
                raise IndexError(f"Feature index {key} out of range")
            return self._features[key]
        else:
            raise TypeError(f"Key must be string or int, got {type(key)}")
    
    def __contains__(self, name: str) -> bool:
        """Support 'in' operator for feature names."""
        return name in self._name_to_feature

    def __len__(self) -> int:
        """Return number of features."""
        return len(self._features)

    @property
    def all_features(self) -> list[str]:
        """Get all feature names."""
        return [f.name for f in self._features]

    def __repr__(self):
        return (
            "FeatureSpec(\n"
            f"  y      : {self.y}\n"
            f"  x      : {self.x}\n"
            f"  sec_x  : {self.sec_x}\n"
            ")"
        )
