import keyword
import warnings
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict
from dataclasses import dataclass, field
from quantbullet.core.enums import DataType
from quantbullet.utils.serialize import to_json_value
    
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype.value if isinstance(self.dtype, Enum) else self.dtype,
            "role": self.role.value if isinstance(self.role, Enum) else self.role,
            "specs": to_json_value(self.specs),
        }

    def to_r_gam_term(self) -> str:
        """Convert feature to R GAM formula term.

        For numeric features with spline specs: s(name, k=n_splines, by=by_var)
        For categorical features: just the name (R treats as factor)
        """
        if self.dtype.is_category():
            return self.name

        # Numeric feature - check if it has spline specs
        if self.specs and "n_splines" in self.specs:
            parts = [self.name]
            k = self.specs.get("n_splines")
            if k is not None:
                parts.append(f"k={k}")
            by = self.specs.get("by")
            if by is not None:
                parts.append(f"by={by}")
            return f"s({', '.join(parts)})"

        # Numeric without spline specs - linear term
        return self.name

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Feature":
        return cls(
            name=data["name"],
            dtype=DataType(data["dtype"]),
            role=FeatureRole(data["role"]),
            specs=data.get("specs") or {},
        )

class FeatureSpec:
    def __init__(self, features: list[Feature]):
        self._features = features
        self._check_duplicate_names()
        self._classify_features()
        self._build_typed_feature_namespace()
        self._build_lookup_maps()

    def _check_duplicate_names(self):
        """Check for duplicate feature names."""
        names = [f.name for f in self._features]
        duplicates = set([name for name in names if names.count(name) > 1])
        if duplicates:
            raise ValueError(f"Duplicate feature names found: {duplicates}")

    def _build_lookup_maps(self):
        """Build direct lookup dictionaries for O(1) access.
        
        _name_to_feature : Dict[str, Feature] - map feature name to Feature object
        _name_to_index   : Dict[str, int]     - map feature name to its index in the original list
        """
        self._name_to_feature = {f.name: f for f in self._features}
        self._name_to_index = {f.name: i for i, f in enumerate(self._features)}

    @property
    def all_inputs( self ) -> list[str]:
        """Get all model input feature names."""
        # there may be duplicates between x and sec_x, use the order in self.x first and then add in order of sec_x
        seen = set()
        all_inputs = []
        for name in self.x + self.sec_x:
            if name not in seen:
                seen.add(name)
                all_inputs.append(name)
        return all_inputs
    
    @property
    def all_inputs_order_map( self ) -> Dict[str, int]:
        """Get a map of all input feature names to their order index."""
        return { name: idx for idx, name in enumerate( self.all_inputs ) }

    def _classify_features(self):
        """Classify features into model inputs, target, references, etc."""
        self.x = [f.name for f in self._features if f.role == FeatureRole.MODEL_INPUT]
        self.x_num = [f.name for f in self._features if f.role == FeatureRole.MODEL_INPUT and f.dtype.is_numeric()]
        self.x_cat = [f.name for f in self._features if f.role == FeatureRole.MODEL_INPUT and f.dtype.is_category()]
        self.sec_x = [f.name for f in self._features if f.role == FeatureRole.SECONDARY_INPUT]
        self.sec_x_num = [f.name for f in self._features if f.role == FeatureRole.SECONDARY_INPUT and f.dtype.is_numeric()]
        self.sec_x_cat = [f.name for f in self._features if f.role == FeatureRole.SECONDARY_INPUT and f.dtype.is_category()]
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

    def to_dict(self) -> Dict[str, Any]:
        return {"features": [f.to_dict() for f in self._features]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSpec":
        features = [Feature.from_dict(entry) for entry in data.get("features", [])]
        return cls(features=features)

    def to_r_gam_formula(self) -> str:
        """Export feature spec to R GAM formula string.

        Generates formula like: y ~ s(x1, k=20) + s(x2, k=20, by=group) + cat_var

        Returns
        -------
        str
            R formula string for use with mgcv::gam or mgcv::bam
        """
        # Get model input features only
        input_features = [f for f in self._features if f.role == FeatureRole.MODEL_INPUT]
        terms = [f.to_r_gam_term() for f in input_features]
        rhs = " + ".join(terms)
        return f"{self.y} ~ {rhs}"
