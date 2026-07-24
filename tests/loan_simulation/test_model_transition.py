import pytest

from quantbullet.loan_simulation import (
    CashflowEngine,
    CompositeTransitionModel,
    ConstantRecoveryLagProvider,
    ConstantSeverityProvider,
    FeatureContext,
    Loan,
    LoanSimulator,
    LoanState,
    MatrixPaymentPolicy,
    StatusConfig,
)


def _status_config():
    return StatusConfig(
        valid_statuses={"C", "D1M", "LIQ"},
        terminal_statuses={"LIQ"},
        prepay_statuses=set(),
        default_statuses={"LIQ"},
        delinquency_buckets={"D1M": "dq30_balance"},
    )


def test_composite_transition_model_combines_edges_with_residual_stay():
    """Configured edge probabilities plus residual stay form a full row."""
    model = CompositeTransitionModel(
        {
            "C": {
                "D1M": 0.20,
                "LIQ": lambda context: 0.10,
            },
            "D1M": {},
        },
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")

    probabilities = model.predict(loan, loan.initial_state())

    assert probabilities == {"D1M": 0.20, "LIQ": 0.10, "C": 0.70}


def test_composite_transition_model_passes_feature_context_to_callables():
    """Edge callables receive loan, state, macro features, and path features."""
    captured_contexts = []

    def edge_probability(context: FeatureContext) -> float:
        captured_contexts.append(context)
        return 0.25 if context.macro_features["rate"] > 0.05 else 0.10

    model = CompositeTransitionModel(
        {
            "C": {"D1M": edge_probability},
            "D1M": {},
        },
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")

    probabilities = model.predict(
        loan,
        loan.initial_state(),
        macro_features={"rate": 0.06},
        path_features={"ever_delinquent": False},
    )

    assert probabilities == {"D1M": 0.25, "C": 0.75}
    assert captured_contexts[0].loan is loan
    assert captured_contexts[0].macro_features == {"rate": 0.06}
    assert captured_contexts[0].path_features == {"ever_delinquent": False}


def test_composite_transition_model_requires_all_non_terminal_statuses():
    """Missing non-terminal rows are explicit configuration errors."""
    with pytest.raises(ValueError, match="missing non-terminal statuses"):
        CompositeTransitionModel(
            {"C": {"D1M": 0.20}},
            status_config=_status_config(),
        )


def test_composite_transition_model_rejects_terminal_edges():
    """Terminal statuses are automatic self-loops and cannot define edges."""
    with pytest.raises(ValueError, match="Terminal status"):
        CompositeTransitionModel(
            {
                "C": {},
                "D1M": {},
                "LIQ": {"C": 0.10},
            },
            status_config=_status_config(),
        )


def test_composite_transition_model_rejects_explicit_stay_edges():
    """Stay probability is residual and should not be configured as an edge."""
    with pytest.raises(ValueError, match="Stay probability"):
        CompositeTransitionModel(
            {
                "C": {"C": 0.10},
                "D1M": {},
            },
            status_config=_status_config(),
        )


def test_composite_transition_model_rejects_probability_sum_above_one():
    """Runtime row sums above one fail with loan and period context."""
    model = CompositeTransitionModel(
        {
            "C": {"D1M": 0.80, "LIQ": 0.30},
            "D1M": {},
        },
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")

    with pytest.raises(ValueError, match="loan_id='L1'.*period=0"):
        model.predict(loan, loan.initial_state())


def test_composite_transition_model_self_loops_terminal_statuses():
    """Terminal states predict a self-loop even though they are not configured."""
    model = CompositeTransitionModel(
        {
            "C": {},
            "D1M": {},
        },
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")
    terminal_state = LoanState("L1", period=3, age_months=3, balance=0.0, status="LIQ")

    assert model.predict(loan, terminal_state) == {"LIQ": 1.0}


def test_composite_transition_model_reports_invalid_callable_probabilities():
    """Callable edge probabilities are validated at prediction time."""
    model = CompositeTransitionModel(
        {
            "C": {"D1M": lambda context: 1.50},
            "D1M": {},
        },
        status_config=_status_config(),
    )
    loan = Loan("L1", 1000.0, 0.12, 12, status="C")

    with pytest.raises(ValueError, match="from_status='C'.*to_status='D1M'"):
        model.predict(loan, loan.initial_state())


def test_composite_transition_model_runs_inside_loan_simulator_with_path_features():
    """Composite transition callables can use path features maintained by the simulator."""
    config = StatusConfig(
        valid_statuses={"C", "D1M"},
        terminal_statuses=set(),
        prepay_statuses=set(),
        default_statuses=set(),
        delinquency_buckets={"D1M": "dq30_balance"},
    )
    model = CompositeTransitionModel(
        {
            "C": {"D1M": lambda context: 1.0},
            "D1M": {"C": lambda context: 1.0 if context.path_features["ever_delinquent"] else 0.0},
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
