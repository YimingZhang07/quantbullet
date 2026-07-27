from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, ClassVar, Literal

import pandas as pd

from .entities import Loan, LoanState

FeatureAdvance = Callable[[Any, Mapping[str, Any]], Any]
FeatureKind = Literal[
    "static_model",
    "dynamic_model",
    "static_context",
    "dynamic_context",
]


@dataclass(frozen=True)
class FeatureSpec:
    """Schema metadata for one runtime feature-state field."""

    name: str
    kind: FeatureKind
    deps: tuple[str, ...]
    advance: FeatureAdvance | None = None
    carry_forward: bool = False


def build_feature_specs(feature_state_type: type[Any]) -> tuple[FeatureSpec, ...]:
    """Build feature specs from dataclass field metadata."""
    if not is_dataclass(feature_state_type):
        raise TypeError("feature_state_type must be a dataclass type")

    specs = []
    for dataclass_field in fields(feature_state_type):
        metadata = dataclass_field.metadata
        specs.append(
            FeatureSpec(
                name=dataclass_field.name,
                kind=metadata["kind"],
                deps=tuple(metadata.get("deps", ())),
                advance=metadata.get("advance"),
                carry_forward=bool(metadata.get("carry_forward", False)),
            )
        )
    return tuple(specs)


class FeatureStateBase(ABC):
    """Base class for dataclass-backed runtime feature states."""

    FEATURE_SPECS: ClassVar[tuple[FeatureSpec, ...]] = ()
    MODEL_FEATURE_NAMES: ClassVar[tuple[str, ...]] = ()
    ADVANCED_FEATURE_NAMES: ClassVar[tuple[str, ...]] = ()
    PROVIDER_FEATURE_NAMES: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def configure_feature_metadata(cls) -> None:
        specs = build_feature_specs(cls)
        cls.FEATURE_SPECS = specs
        cls.MODEL_FEATURE_NAMES = tuple(
            spec.name
            for spec in specs
            if spec.kind in {"static_model", "dynamic_model"}
        )
        cls.ADVANCED_FEATURE_NAMES = tuple(
            spec.name for spec in specs if spec.advance is not None
        )
        cls.PROVIDER_FEATURE_NAMES = tuple(
            spec.name
            for spec in specs
            if spec.kind in {"static_context", "dynamic_context"}
        )

    def model_features(self) -> Mapping[str, Any]:
        return {name: getattr(self, name) for name in self.MODEL_FEATURE_NAMES}


class RuntimeFeatureProvider(ABC):
    """Base hook for per-path model feature state managed by ``LoanSimulator``."""

    def initialize_path_state(self, loan: Loan, start_period: pd.Period) -> Any:
        """Create per-path feature state at path start."""
        return None

    def prepare_period_state(
        self,
        *,
        loan: Loan,
        current_state: LoanState,
        period_date: pd.Period,
        macro_features: Mapping[str, Any],
        path_features: Mapping[str, Any],
        feature_state: Any,
    ) -> None:
        """Update feature state before model features are read for a period."""
        return

    def model_features_for_period(
        self,
        *,
        loan: Loan,
        current_state: LoanState,
        period_date: pd.Period,
        macro_features: Mapping[str, Any],
        path_features: Mapping[str, Any],
        feature_state: Any,
    ) -> Mapping[str, Any]:
        """Return model-ready features for the current transition period."""
        return {}


class EmptyRuntimeFeatureProvider(RuntimeFeatureProvider):
    """Default provider for simulations that do not need runtime model features."""
