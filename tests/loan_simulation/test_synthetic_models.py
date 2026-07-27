import sys
from pathlib import Path

import pandas as pd
import pytest

from quantbullet.loan_simulation import FeatureContext, Loan, LoanState


SYNTHETIC_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "loan_simulation"
    / "examples"
    / "synthetic"
)
sys.path.insert(0, str(SYNTHETIC_DIR))

from feature_builder import (  # noqa: E402
    MODEL_FEATURE_NAMES,
    SyntheticFeatureProvider,
    build_feature_dict,
)
from models import (  # noqa: E402
    CHARGED_OFF,
    CURRENT,
    C_TO_D1_MODEL,
    C_TO_PIF_MODEL,
    DELINQUENT_1,
    DELINQUENT_2,
    PREPAID,
    build_status_config,
    build_transition_model,
)


def _context(
    *,
    age: float = 20.0,
    incentive: float = 0.02,
    hpi: float = 110.0,
) -> FeatureContext:
    loan = Loan(
        "L1",
        balance=100_000.0,
        annual_rate=0.09,
        term_months=120,
        status=CURRENT,
    )
    state = LoanState(
        "L1",
        period=int(age),
        age_months=int(age),
        balance=100_000.0,
        status=CURRENT,
    )
    return FeatureContext(
        loan=loan,
        current_state=state,
        model_features={
            "age": age,
            "incentive": incentive,
            "hpi": hpi,
        },
    )


def test_synthetic_feature_provider_builds_only_declared_features():
    loan = Loan("L1", 100_000.0, 0.09, 120, status=CURRENT)
    state = LoanState("L1", period=12, age_months=12, balance=90_000.0, status=CURRENT)
    provider = SyntheticFeatureProvider()

    features = provider.model_features_for_period(
        loan=loan,
        current_state=state,
        period_date=pd.Period("2001-01", freq="M"),
        macro_features={"market_rate": 0.06, "hpi": 105.0, "ignored": 1.0},
        path_features={"ignored": True},
        feature_state=None,
    )

    assert tuple(features) == MODEL_FEATURE_NAMES
    assert features == {"age": 12.0, "incentive": 0.03, "hpi": 105.0}


def test_build_feature_dict_filters_to_model_contract():
    context = _context()
    context = FeatureContext(
        loan=context.loan,
        current_state=context.current_state,
        model_features={**context.model_features, "ignored": 999.0},
    )

    assert build_feature_dict(context) == {
        "age": 20.0,
        "incentive": 0.02,
        "hpi": 110.0,
    }


def test_dynamic_probability_models_respond_to_their_three_features():
    base = _context()

    assert C_TO_D1_MODEL(_context(age=40.0)) > C_TO_D1_MODEL(base)
    assert C_TO_D1_MODEL(_context(incentive=0.03)) > C_TO_D1_MODEL(base)
    assert C_TO_D1_MODEL(_context(hpi=120.0)) < C_TO_D1_MODEL(base)

    assert C_TO_PIF_MODEL(_context(age=40.0)) > C_TO_PIF_MODEL(base)
    assert C_TO_PIF_MODEL(_context(incentive=0.03)) > C_TO_PIF_MODEL(base)
    assert C_TO_PIF_MODEL(_context(hpi=120.0)) > C_TO_PIF_MODEL(base)


def test_dynamic_probability_models_apply_declared_bounds():
    d1_high = _context(age=1_000.0, incentive=1.0, hpi=-1_000.0)
    d1_low = _context(age=0.0, incentive=-1.0, hpi=1_000.0)
    pif_high = _context(age=1_000.0, incentive=1.0, hpi=1_000.0)
    pif_low = _context(age=0.0, incentive=-1.0, hpi=-1_000.0)

    assert C_TO_D1_MODEL(d1_high) == pytest.approx(C_TO_D1_MODEL.maximum)
    assert C_TO_D1_MODEL(d1_low) == pytest.approx(C_TO_D1_MODEL.minimum)
    assert C_TO_PIF_MODEL(pif_high) == pytest.approx(C_TO_PIF_MODEL.maximum)
    assert C_TO_PIF_MODEL(pif_low) == pytest.approx(C_TO_PIF_MODEL.minimum)


def test_current_row_has_no_direct_charge_off_and_uses_odds_normalization():
    context = _context()
    model = build_transition_model(build_status_config())

    probabilities = model.predict(
        context.loan,
        context.current_state,
        model_features=context.model_features,
    )

    d1_probability = C_TO_D1_MODEL(context)
    pif_probability = C_TO_PIF_MODEL(context)
    d1_odds = d1_probability / (1.0 - d1_probability)
    pif_odds = pif_probability / (1.0 - pif_probability)
    denominator = 1.0 + d1_odds + pif_odds

    assert set(model.probabilities[CURRENT]) == {DELINQUENT_1, PREPAID}
    assert CHARGED_OFF not in probabilities
    assert probabilities[DELINQUENT_1] == pytest.approx(d1_odds / denominator)
    assert probabilities[PREPAID] == pytest.approx(pif_odds / denominator)
    assert probabilities[CURRENT] == pytest.approx(1.0 / denominator)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_delinquent_rows_use_constant_competing_probabilities():
    model = build_transition_model(build_status_config())

    assert model.probabilities[DELINQUENT_1] == {
        CURRENT: 0.30,
        DELINQUENT_2: 0.25,
        PREPAID: 0.03,
        CHARGED_OFF: 0.02,
    }
    assert model.probabilities[DELINQUENT_2] == {
        CURRENT: 0.10,
        DELINQUENT_1: 0.20,
        PREPAID: 0.02,
        CHARGED_OFF: 0.25,
    }


@pytest.mark.parametrize("terminal_status", [PREPAID, CHARGED_OFF])
def test_terminal_states_are_automatic_self_loops(terminal_status):
    model = build_transition_model(build_status_config())
    loan = Loan("L1", 100_000.0, 0.09, 120, status=CURRENT)
    state = LoanState(
        "L1",
        period=3,
        age_months=3,
        balance=0.0,
        status=terminal_status,
    )

    assert model.predict(loan, state) == {terminal_status: 1.0}


def test_normal_paydown_is_not_a_terminal_status():
    status_config = build_status_config()

    assert status_config.terminal_statuses == {PREPAID, CHARGED_OFF}
