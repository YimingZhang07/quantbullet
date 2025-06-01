import keyword
from enum import Enum
from types import SimpleNamespace
from dataclasses import dataclass
from quantbullet.core.enums import DataType
    
class FeatureRole(Enum):
    MODEL_INPUT         = "model_input"
    TARGET              = "target"
    AUXILIARY_TARGET    = "auxiliary_target"
    REFERENCE           = "reference"
    GROUPING            = "grouping"

@dataclass
class Feature:
    name    : str
    dtype   : DataType
    role    : FeatureRole

class FeatureSpec:
    def __init__(self, features: list[Feature]):
        self._features = features
        self._classify_features()
        self._build_feature_namespace()

    def _classify_features(self):
        """Classify features into model inputs, target, references, etc."""
        self.x = [f.name for f in self._features if f.role == FeatureRole.MODEL_INPUT]
        self.x_num = [f.name for f in self._features if f.role == FeatureRole.MODEL_INPUT and f.dtype.is_numeric()]
        self.x_cat = [f.name for f in self._features if f.role == FeatureRole.MODEL_INPUT and f.dtype.is_categorical()]
        self.refs = [f.name for f in self._features if f.role == FeatureRole.REFERENCE]
        
        # Expect exactly one target
        targets = [f.name for f in self._features if f.role == FeatureRole.TARGET]
        if len(targets) != 1:
            raise ValueError(f"Expected exactly one target feature, got: {targets}")
        self.y = targets[0]
        
    def _build_feature_namespace(self):
        """Build a namespace for enum-style access to feature names."""
        fields = {}
        for f in self._features:
            key = f.name.upper()
            if not key.isidentifier() or keyword.iskeyword(key):
                raise ValueError(f"Invalid feature name for enum-style access: {f.name}")
            fields[key] = f.name
        self.FEATURES = SimpleNamespace(**fields)

    @property
    def all_features(self) -> list[str]:
        """Get all feature names."""
        return [f.name for f in self._features]

    def __repr__(self):
        return (
            "FeatureSpec(\n"
            f"  y      : {self.y}\n"
            f"  x      : {self.x}\n"
            f"  x_num  : {self.x_num}\n"
            f"  x_cat  : {self.x_cat}\n"
            f"  refs   : {self.refs}\n"
            ")"
        )
