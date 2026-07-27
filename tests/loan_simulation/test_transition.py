import random

import pytest

from quantbullet.loan_simulation import (
    ConstantTransitionModel,
    Loan,
    StatusConfig,
    sample_next_status,
)


def test_custom_status_config_supports_roll_rate_style_states():
    config = StatusConfig(
        valid_statuses={"C", "D1M", "PIF", "LIQ"},
        terminal_statuses={"PIF", "LIQ"},
        prepay_statuses={"PIF"},
        default_statuses={"LIQ"},
        delinquency_buckets={"D1M": "dq30_balance"},
    )

    assert config.is_terminal("PIF")
    assert config.is_prepay("PIF")
    assert config.is_default("LIQ")
    assert config.delinquency_bucket("D1M") == "dq30_balance"


def test_status_config_requires_explicit_status_vocabulary():
    with pytest.raises(TypeError, match="valid_statuses"):
        StatusConfig()


def test_status_config_rejects_statuses_outside_vocabulary():
    with pytest.raises(ValueError, match="not in valid_statuses"):
        StatusConfig(
            valid_statuses={"C", "PIF"},
            terminal_statuses={"PIF", "LIQ"},
            prepay_statuses={"PIF"},
            default_statuses={"LIQ"},
            delinquency_buckets={},
        )


def test_constant_transition_model_predicts_configured_probabilities():
    config = StatusConfig(
        valid_statuses={"C", "D1M", "PIF", "LIQ"},
        terminal_statuses={"PIF", "LIQ"},
        prepay_statuses={"PIF"},
        default_statuses={"LIQ"},
        delinquency_buckets={"D1M": "dq30_balance"},
    )
    model = ConstantTransitionModel(
        {
            "C": {"C": 0.90, "D1M": 0.05, "PIF": 0.03, "LIQ": 0.02},
            "D1M": {"C": 0.20, "D1M": 0.70, "PIF": 0.05, "LIQ": 0.05},
            "PIF": {"PIF": 1.0},
            "LIQ": {"LIQ": 1.0},
        },
        status_config=config,
    )
    loan = Loan("L1", balance=1000.0, annual_rate=0.12, term_months=12, status="C")

    probabilities = model.predict(loan, loan.initial_state())

    assert probabilities == {"C": 0.90, "D1M": 0.05, "PIF": 0.03, "LIQ": 0.02}


def test_constant_transition_model_requires_complete_status_coverage():
    config = StatusConfig(
        valid_statuses={"C", "PIF"},
        terminal_statuses={"PIF"},
        prepay_statuses={"PIF"},
        default_statuses=set(),
        delinquency_buckets={},
    )

    with pytest.raises(ValueError, match="missing rows"):
        ConstantTransitionModel(
            {"C": {"C": 0.90, "PIF": 0.10}},
            status_config=config,
        )


@pytest.mark.parametrize(
    "transitions,match",
    [
        ({"C": {"C": 0.90, "PIF": 0.10}, "BAD": {"BAD": 1.0}}, "Unknown status"),
        ({"C": {"C": 0.90, "PIF": 0.20}, "PIF": {"PIF": 1.0}}, "sum to 1.0"),
        ({"C": {"C": 1.10, "PIF": -0.10}, "PIF": {"PIF": 1.0}}, "cannot be negative"),
    ],
)
def test_constant_transition_model_rejects_invalid_tables(transitions, match):
    config = StatusConfig(
        valid_statuses={"C", "PIF"},
        terminal_statuses={"PIF"},
        prepay_statuses={"PIF"},
        default_statuses=set(),
        delinquency_buckets={},
    )

    with pytest.raises(ValueError, match=match):
        ConstantTransitionModel(transitions, status_config=config)


def test_sample_next_status_is_reproducible_with_seeded_rng():
    probabilities = {"C": 0.80, "D1M": 0.20}

    first = sample_next_status(probabilities, random.Random(1))
    second = sample_next_status(probabilities, random.Random(1))

    assert first == second == "C"
