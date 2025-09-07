import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from quantbullet.parametic_model import (
    DoubleLogisticModel,
    AsymQuadModel,
    InterpolatedModel,
    SigmoidModel,
    BathtubModel,
)

MODEL_REGISTRY = {
    "DoubleLogisticModel": DoubleLogisticModel,
    "AsymQuadModel": AsymQuadModel,
    "InterpolatedModel": InterpolatedModel,
    "SigmoidModel": SigmoidModel,
    "BathtubModel": BathtubModel,
}

@dataclass
class ComponentConfig:
    class_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def build( self):
        cls = MODEL_REGISTRY[self.class_name]
        return cls.from_dict( self.args )

@dataclass
class ComponentRegistry:
    label: str = field(default="CompositeModel")   # safer default
    components: Dict[str, ComponentConfig] = field(default_factory=dict)

    def items(self):
        return self.components.items()

def dataclass_to_dict(model_config: ComponentRegistry) -> dict:
    return asdict(model_config)

def dict_to_dataclass(d: Dict[str, Any]) -> ComponentRegistry:
    """Convert dict → ComponentRegistry with nested ComponentConfig."""
    components = {
        name: ComponentConfig(**config)
        for name, config in d["components"].items()
    }
    return ComponentRegistry(
        label=d.get("label", "CompositeModel"),
        components=components
    )

class ComponentManager:
    __slots__ = ['registry', 'models']

    def __init__(self, registry: ComponentRegistry = None):
        self.registry = registry if registry is not None else ComponentRegistry()

    @property
    def component_names(self) -> List[str]:
        return list(self.registry.components.keys())

    # ---- CRUD ----
    def add_component(self, name: str, component: ComponentConfig):
        self.registry.components[name] = component

    def remove_component(self, name: str):
        if name in self.registry.components:
            del self.registry.components[name]

    def get_component(self, name: str) -> ComponentConfig:
        return self.registry.components[name]

    # ---- Serialization ----
    def to_dict(self) -> Dict:
        return asdict(self.registry)

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(dict_to_dataclass(d))

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    def build_all_models(self):
        self.models = {name: component.build() for name, component in self.registry.items()}

    def __repr__(self):
        return f"ComponentManager(components={self.component_names})"

    def __getitem__(self, name: str):
        return self.models[name]