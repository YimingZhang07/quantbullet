from dataclasses import dataclass
from typing import Any

import pytest

from quantbullet.loan_simulation import (
    MISSING,
    FeatureStateBase,
    FeatureUpdateError,
    model_feature,
)


def update_score(state: Any, env: Any) -> float:
    return state.base_score + env["lift"]


def update_lookup(state: Any, env: Any) -> Any:
    return env.get("lookup", MISSING)


@dataclass
class ExampleFeatureState(FeatureStateBase):
    base_score: float
    score: float = model_feature(update_score, deps=("base_score",), init=False)
    grade: str = model_feature(default="A")
    lookup: float = model_feature(update_lookup, carry_forward=True, default=1.5)


def test_model_feature_fields_define_schema_and_update_logic():
    state = ExampleFeatureState(base_score=1.0)

    state.update_features({"lift": 2.0})

    assert state.model_feature_names == ("score", "grade", "lookup")
    assert state.feature_specs[0].deps == ("base_score",)
    assert state.model_features() == {"score": 3.0, "grade": "A", "lookup": 1.5}


def test_carry_forward_keeps_previous_value_on_missing():
    state = ExampleFeatureState(base_score=1.0)

    state.update_features({"lift": 0.0, "lookup": 2.5})
    state.update_features({"lift": 0.0})

    assert state.lookup == 2.5


def test_missing_without_carry_forward_raises():
    @dataclass
    class StrictFeatureState(FeatureStateBase):
        value: float = model_feature(lambda state, env: MISSING, default=0.0)

    with pytest.raises(FeatureUpdateError, match="'value'"):
        StrictFeatureState().update_features({})


def test_feature_specs_are_cached_per_class_not_per_instance():
    first = ExampleFeatureState(base_score=1.0)
    second = ExampleFeatureState(base_score=2.0)

    assert first.feature_specs is second.feature_specs
