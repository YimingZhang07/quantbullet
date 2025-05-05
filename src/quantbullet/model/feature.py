from enum import Enum
from dataclasses import dataclass
from quantbullet.core.types import DataType
    
class FeatureRole(Enum):
    MODEL_INPUT = "model_input"
    TARGET = "target"
    AUXILIARY_TARGET = "auxiliary_target"
    REFERENCE = "reference"
    GROUPING = "grouping"

@dataclass
class Feature:
    name: str
    dtype: DataType
    role: FeatureRole

class FeatureSpec:
    def __init__(self, features: list[Feature]):
        self.features = features
        self._build_feature_groups()

    def _build_feature_groups(self):
        self.x = [f.name for f in self.features if f.role == FeatureRole.MODEL_INPUT]
        self.x_num = [f.name for f in self.features if f.role == FeatureRole.MODEL_INPUT and f.dtype.is_numeric()]
        self.x_cat = [f.name for f in self.features if f.role == FeatureRole.MODEL_INPUT and f.dtype.is_categorical()]
        self.refs = [f.name for f in self.features if f.role == FeatureRole.REFERENCE]
        
        # Expect exactly one target
        targets = [f.name for f in self.features if f.role == FeatureRole.TARGET]
        if len(targets) != 1:
            raise ValueError(f"Expected exactly one target feature, got: {targets}")
        self.y = targets[0]

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
