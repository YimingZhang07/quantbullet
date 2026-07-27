from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cache
from typing import Any

import pandas as pd

from .entities import Loan, LoanState

_FEATURE_METADATA_KEY = "quantbullet.loan_simulation.feature"

# Sentinel returned by feature updates that cannot produce a value this period.
MISSING: Any = object()

FeatureUpdate = Callable[[Any, Mapping[str, Any]], Any]


class FeatureUpdateError(ValueError):
    """Raised when a feature update returns MISSING and cannot carry forward."""


@dataclass(frozen=True)
class RuntimeFeatureSpec:
    """Schema entry derived from one ``model_feature`` field declaration."""

    name: str
    update: FeatureUpdate | None = None
    carry_forward: bool = False
    deps: tuple[str, ...] = ()


def model_feature(
    update: FeatureUpdate | None = None,
    *,
    carry_forward: bool = False,
    deps: tuple[str, ...] = (),
    **field_kwargs: Any,
) -> Any:
    """Declare a model-facing feature field on a feature-state dataclass.

    The dataclass field itself is the single source of truth: its name, model
    membership, and update logic all live on the declaration. Fields declared
    without this helper are provider context and are never sent to models.

    ``update`` receives ``(state, env)`` and returns the new value, or
    ``MISSING`` when no value can be computed this period. ``carry_forward``
    keeps the previous value on MISSING instead of raising. Extra keyword
    arguments (``default``, ``init``, ...) pass through to ``dataclasses.field``.
    """
    return field(
        metadata={
            _FEATURE_METADATA_KEY: {
                "update": update,
                "carry_forward": carry_forward,
                "deps": tuple(deps),
            }
        },
        **field_kwargs,
    )


@cache
def _feature_specs(feature_state_type: type) -> tuple[RuntimeFeatureSpec, ...]:
    """Read model_feature declarations from a dataclass, once per class."""
    if not is_dataclass(feature_state_type):
        raise TypeError(
            "Feature states must be dataclasses; "
            f"{feature_state_type.__name__} is not"
        )
    specs = []
    for dataclass_field in fields(feature_state_type):
        declaration = dataclass_field.metadata.get(_FEATURE_METADATA_KEY)
        if declaration is None:
            continue
        specs.append(RuntimeFeatureSpec(name=dataclass_field.name, **declaration))
    return tuple(specs)


class FeatureStateBase:
    """Base class for dataclass feature states built from model_feature fields.

    The schema is introspected lazily on first use and cached once per class,
    so subclasses need no registration call and instances share one schema.
    """

    @property
    def feature_specs(self) -> tuple[RuntimeFeatureSpec, ...]:
        return _feature_specs(type(self))

    @property
    def model_feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.feature_specs)

    def update_features(self, env: Mapping[str, Any]) -> None:
        """Run each feature's own update logic in declaration order."""
        for spec in self.feature_specs:
            if spec.update is None:
                continue
            value = spec.update(self, env)
            if value is MISSING:
                if spec.carry_forward:
                    continue
                raise FeatureUpdateError(
                    f"Feature {spec.name!r} returned MISSING and is not carry-forward"
                )
            setattr(self, spec.name, value)

    def model_features(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.model_feature_names}


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
