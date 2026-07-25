import math

import numpy as np
import pytest

from quantbullet.loan_simulation import Loan, StatusConfig
from quantbullet.loan_simulation.adapters import build_softmax_transition_model
from quantbullet.model.gam.terms import SplineTermData
from quantbullet.model.gam_replay import GAMReplayModel


def _linear_replay_model(offset: float) -> GAMReplayModel:
    x_values = np.asarray([0.0, 1.0, 2.0, 3.0])
    return GAMReplayModel(
        term_data={
            "x": SplineTermData(
                feature="x",
                x=x_values,
                y=x_values + offset,
                interpolation="linear",
            )
        }
    )


def _status_config():
    return StatusConfig(
        valid_statuses={"C", "D1M", "PIF"},
        terminal_statuses={"PIF"},
        prepay_statuses={"PIF"},
        default_statuses=set(),
        delinquency_buckets={"D1M": "dq30_balance"},
    )


def test_build_softmax_transition_model_wraps_replay_edges():
    """Replay edge models become FeatureContext-driven softmax logits."""
    captured_contexts = []

    def feature_builder(context):
        captured_contexts.append(context)
        return {"x": context.loan.metadata["x"]}

    model = build_softmax_transition_model(
        {
            "C": {
                "PIF": _linear_replay_model(1.0),
                "D1M": _linear_replay_model(-1.0),
            }
        },
        feature_builder=feature_builder,
        status_config=_status_config(),
    )
    loan = Loan("L1", 1_000.0, 0.12, 12, status="C", metadata={"x": 2.0})

    probabilities = model.predict(loan, loan.initial_state())

    denominator = 1.0 + math.exp(3.0) + math.exp(1.0)
    assert probabilities["PIF"] == pytest.approx(math.exp(3.0) / denominator)
    assert probabilities["D1M"] == pytest.approx(math.exp(1.0) / denominator)
    assert probabilities["C"] == pytest.approx(1.0 / denominator)
    assert captured_contexts[0].loan is loan


def test_build_softmax_transition_model_fills_missing_non_terminal_rows():
    """Missing non-terminal source statuses are explicit pure-stay rows."""

    def feature_builder(context):
        return {"x": context.loan.metadata["x"]}

    model = build_softmax_transition_model(
        {"C": {"PIF": _linear_replay_model(0.0)}},
        feature_builder=feature_builder,
        status_config=_status_config(),
    )
    loan = Loan("L1", 1_000.0, 0.12, 12, status="D1M", metadata={"x": 2.0})

    assert model.predict(loan, loan.initial_state()) == {"D1M": 1.0}


def test_build_softmax_transition_model_rejects_terminal_source_rows():
    """Terminal statuses self-loop and cannot define replay edge models."""

    def feature_builder(context):
        return {"x": context.loan.metadata["x"]}

    with pytest.raises(ValueError, match="Terminal status"):
        build_softmax_transition_model(
            {"PIF": {"C": _linear_replay_model(0.0)}},
            feature_builder=feature_builder,
            status_config=_status_config(),
        )


def test_build_softmax_transition_model_rejects_non_gam_replay_models():
    """The bridge intentionally accepts only GAMReplayModel edge objects."""

    def feature_builder(context):
        return {"x": context.loan.metadata["x"]}

    with pytest.raises(TypeError, match="GAMReplayModel"):
        build_softmax_transition_model(
            {"C": {"PIF": object()}},
            feature_builder=feature_builder,
            status_config=_status_config(),
        )
