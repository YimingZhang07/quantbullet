import math

import pytest

from quantbullet.loan_simulation import (
    CashflowEngine,
    ConstantRecoveryLagProvider,
    ConstantSeverityProvider,
    FeatureContext,
    Loan,
    LoanSimulator,
    LoanState,
    MatrixPaymentPolicy,
    ProbabilitySoftmaxTransitionModel,
    SoftmaxTransitionModel,
    StatusConfig,
)


def _status_config():
    return StatusConfig(
        valid_statuses={"C", "D1M", "PIF"},
        terminal_statuses={"PIF"},
        prepay_statuses={"PIF"},
        default_statuses=set(),
        delinquency_buckets={"D1M": "dq30_balance"},
    )


def test_softmax_transition_model_matches_hand_computed_probabilities():
    """Constant logits produce the documented stay-based softmax row."""
    model = SoftmaxTransitionModel(
        {
            "C": {"PIF": 1.0, "D1M": -1.0},
            "D1M": {},
        },
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")

    probabilities = model.predict(loan, loan.initial_state())

    denominator = 1.0 + math.exp(1.0) + math.exp(-1.0)
    assert probabilities["PIF"] == pytest.approx(math.exp(1.0) / denominator)
    assert probabilities["D1M"] == pytest.approx(math.exp(-1.0) / denominator)
    assert probabilities["C"] == pytest.approx(1.0 / denominator)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_softmax_transition_model_empty_row_is_pure_stay():
    """An explicit empty logit row keeps the loan in its current status."""
    model = SoftmaxTransitionModel(
        {"C": {}, "D1M": {}},
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")

    assert model.predict(loan, loan.initial_state()) == {"C": 1.0}


def test_softmax_transition_model_passes_feature_context_to_callables():
    """Logit callables receive loan, state, macro features, and path features."""
    captured_contexts = []

    def pif_logit(context: FeatureContext) -> float:
        captured_contexts.append(context)
        return 2.0 if context.macro_features["market_rate"] < 0.07 else -2.0

    model = SoftmaxTransitionModel(
        {"C": {"PIF": pif_logit}, "D1M": {}},
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")

    probabilities = model.predict(
        loan,
        loan.initial_state(),
        macro_features={"market_rate": 0.06},
        path_features={"ever_delinquent": True},
    )

    denominator = 1.0 + math.exp(2.0)
    assert probabilities["PIF"] == pytest.approx(math.exp(2.0) / denominator)
    assert probabilities["C"] == pytest.approx(1.0 / denominator)
    assert captured_contexts[0].loan is loan
    assert captured_contexts[0].macro_features == {"market_rate": 0.06}
    assert captured_contexts[0].path_features == {"ever_delinquent": True}


def test_softmax_transition_model_self_loops_terminal_statuses():
    """Terminal states predict a self-loop even though they are not configured."""
    model = SoftmaxTransitionModel(
        {"C": {}, "D1M": {}},
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")
    terminal_state = LoanState("L1", period=2, age_months=2, balance=0.0, status="PIF")

    assert model.predict(loan, terminal_state) == {"PIF": 1.0}


def test_softmax_transition_model_requires_all_non_terminal_statuses():
    """Missing non-terminal rows are explicit configuration errors."""
    with pytest.raises(ValueError, match="missing non-terminal statuses"):
        SoftmaxTransitionModel({"C": {}}, status_config=_status_config())


def test_softmax_transition_model_rejects_terminal_logit_rows():
    """Terminal statuses are automatic self-loops and cannot define logits."""
    with pytest.raises(ValueError, match="Terminal status"):
        SoftmaxTransitionModel(
            {"C": {}, "D1M": {}, "PIF": {"C": 0.0}},
            status_config=_status_config(),
        )


def test_softmax_transition_model_rejects_explicit_stay_logits():
    """Stay is the base category with score 0 and cannot be configured."""
    with pytest.raises(ValueError, match="Stay logit"):
        SoftmaxTransitionModel(
            {"C": {"C": 1.0}, "D1M": {}},
            status_config=_status_config(),
        )


def test_softmax_transition_model_rejects_non_finite_constant_logits():
    """Constant logits are validated at construction time."""
    with pytest.raises(ValueError, match="must be finite"):
        SoftmaxTransitionModel(
            {"C": {"PIF": float("inf")}, "D1M": {}},
            status_config=_status_config(),
        )


def test_softmax_transition_model_reports_non_finite_callable_logits():
    """Runtime logit failures carry loan and period context."""
    model = SoftmaxTransitionModel(
        {"C": {"PIF": lambda context: float("nan")}, "D1M": {}},
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")

    with pytest.raises(ValueError, match="loan_id='L1'.*period=0"):
        model.predict(loan, loan.initial_state())


def test_softmax_transition_model_handles_large_logits_without_overflow():
    """Max-shifted softmax keeps huge finite logits from overflowing math.exp."""
    model = SoftmaxTransitionModel(
        {"C": {"PIF": 1000.0}, "D1M": {}},
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")

    probabilities = model.predict(loan, loan.initial_state())

    assert probabilities["PIF"] == pytest.approx(1.0)
    assert probabilities["C"] == pytest.approx(0.0)


def test_softmax_transition_model_runs_inside_loan_simulator_with_path_features():
    """Logit callables can read path features maintained by the simulator."""
    config = StatusConfig(
        valid_statuses={"C", "D1M"},
        terminal_statuses=set(),
        prepay_statuses=set(),
        default_statuses=set(),
        delinquency_buckets={"D1M": "dq30_balance"},
    )
    model = SoftmaxTransitionModel(
        {
            "C": {"D1M": lambda context: 50.0},
            "D1M": {
                "C": lambda context: 50.0
                if context.path_features["ever_delinquent"]
                else -50.0
            },
        },
        status_config=config,
    )
    payment_policy = MatrixPaymentPolicy(
        {
            "C": {"C": 1, "D1M": 0},
            "D1M": {"C": 2, "D1M": 1},
        },
        status_config=config,
    )
    simulator = LoanSimulator(
        model,
        CashflowEngine(
            payment_policy,
            ConstantSeverityProvider(0.40),
            ConstantRecoveryLagProvider(0),
            status_config=config,
        ),
        horizon=2,
        start_date="2026-01-31",
    )

    frame = simulator.simulate_loan(Loan("L1", 1200.0, 0.12, 12, status="C")).to_frame()

    assert frame["end_status"].tolist() == ["D1M", "C"]


def test_probability_softmax_transition_model_competes_binary_probabilities():
    """Independent binary probabilities are converted to odds and normalized."""
    model = ProbabilitySoftmaxTransitionModel(
        {
            "C": {"PIF": 0.90, "D1M": 0.90},
            "D1M": {},
        },
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")

    probabilities = model.predict(loan, loan.initial_state())

    assert probabilities["PIF"] == pytest.approx(9.0 / 19.0)
    assert probabilities["D1M"] == pytest.approx(9.0 / 19.0)
    assert probabilities["C"] == pytest.approx(1.0 / 19.0)


def test_probability_softmax_transition_model_passes_context_to_callables():
    """Probability callables receive the same FeatureContext as logit callables."""
    captured_contexts = []

    def pif_probability(context: FeatureContext) -> float:
        captured_contexts.append(context)
        return 0.20 if context.macro_features["market_rate"] < 0.07 else 0.05

    model = ProbabilitySoftmaxTransitionModel(
        {
            "C": {"PIF": pif_probability},
            "D1M": {},
        },
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")

    probabilities = model.predict(
        loan,
        loan.initial_state(),
        macro_features={"market_rate": 0.06},
        path_features={"ever_delinquent": False},
    )

    odds = 0.20 / 0.80
    assert probabilities["PIF"] == pytest.approx(odds / (1.0 + odds))
    assert captured_contexts[0].loan is loan
    assert captured_contexts[0].macro_features == {"market_rate": 0.06}


def test_probability_softmax_transition_model_rejects_one_probability():
    """A probability of one creates infinite odds and must be modeled directly."""
    with pytest.raises(ValueError, match="< 1"):
        ProbabilitySoftmaxTransitionModel(
            {"C": {"PIF": 1.0}, "D1M": {}},
            status_config=_status_config(),
        )


def test_probability_softmax_transition_model_reports_invalid_callable_probabilities():
    """Runtime probability failures carry loan and period context."""
    model = ProbabilitySoftmaxTransitionModel(
        {"C": {"PIF": lambda context: 1.0}, "D1M": {}},
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")

    with pytest.raises(ValueError, match="loan_id='L1'.*period=0"):
        model.predict(loan, loan.initial_state())
