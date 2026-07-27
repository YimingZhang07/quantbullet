from dataclasses import dataclass, field

from quantbullet.loan_simulation import FeatureStateBase


def test_feature_state_base_derives_schema_metadata():
    def advance_score(state, env):
        return state.base_score + env["lift"]

    @dataclass
    class ExampleFeatureState(FeatureStateBase):
        score: float = field(
            default=0.0,
            metadata={
                "kind": "dynamic_model",
                "deps": ("base_score", "lift"),
                "advance": advance_score,
            },
        )
        base_score: float = field(
            default=1.0,
            metadata={"kind": "static_context"},
        )

    ExampleFeatureState.configure_feature_metadata()
    state = ExampleFeatureState()

    assert ExampleFeatureState.MODEL_FEATURE_NAMES == ("score",)
    assert ExampleFeatureState.ADVANCED_FEATURE_NAMES == ("score",)
    assert ExampleFeatureState.PROVIDER_FEATURE_NAMES == ("base_score",)
    assert ExampleFeatureState.FEATURE_SPECS[0].deps == ("base_score", "lift")

    state.score = ExampleFeatureState.FEATURE_SPECS[0].advance(state, {"lift": 2.0})
    assert state.model_features() == {"score": 3.0}
