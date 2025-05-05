from enum import Enum
from dataclasses import dataclass

class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    REFERENCE = "reference"

@dataclass
class Feature:
    name: str
    type: FeatureType

class FeatureSpec:
    def __init__(self, features: list[Feature]):
        self.features = features
        self._build_feature_groups()

    def _build_feature_groups(self):
        self.model_features = [f.name for f in self.features if f.type != FeatureType.REFERENCE]
        self.reference_features = [f.name for f in self.features if f.type == FeatureType.REFERENCE]
        self.categorical_features = [f.name for f in self.features if f.type == FeatureType.CATEGORICAL]
        self.numeric_features = [f.name for f in self.features if f.type == FeatureType.NUMERIC]
        self.all_features = [f.name for f in self.features]

    def __repr__(self):
        return (
            f"FeatureSpec(\n"
            f"  Model: {self.model_features}\n"
            f"  Numeric: {self.numeric_cols}\n"
            f"  Categorical: {self.categorical_cols}\n"
            f"  Reference: {self.reference_cols}\n)"
        )