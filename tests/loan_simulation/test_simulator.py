import pandas as pd

from quantbullet.loan_simulation import (
    CashflowEngine,
    ConstantRecoveryLagProvider,
    ConstantSeverityProvider,
    ConstantTransitionModel,
    DataFrameMacroFeatureProvider,
    Loan,
    LoanSimulationResult,
    LoanSimulator,
    MatrixPaymentPolicy,
    PeriodCashflow,
    PortfolioSimulationResult,
    PortfolioSimulator,
    RuntimeFeatureProvider,
    StatusConfig,
    TransitionModel,
)


def _status_config():
    return StatusConfig(
        valid_statuses={"C", "PIF", "LIQ"},
        terminal_statuses={"PIF", "LIQ"},
        prepay_statuses={"PIF"},
        default_statuses={"LIQ"},
        delinquency_buckets={},
    )


def _cashflow_engine(recovery_lag=3):
    config = _status_config()
    matrix = {
        "C": {"C": 1, "PIF": 0, "LIQ": 0},
        "PIF": {"C": 0, "PIF": 0, "LIQ": 0},
        "LIQ": {"C": 0, "PIF": 0, "LIQ": 0},
    }
    return CashflowEngine(
        MatrixPaymentPolicy(matrix, status_config=config),
        ConstantSeverityProvider(0.40),
        ConstantRecoveryLagProvider(recovery_lag),
        status_config=config,
    )


def test_loan_simulator_passes_macro_features_by_calendar_month():
    """Simulator maps 1-based projection periods to calendar-month macro rows."""
    class RecordingTransitionModel(TransitionModel):
        def __init__(self):
            self.hpi_values = []

        def predict(
            self,
            loan,
            current_state,
            macro_features=None,
            path_features=None,
            model_features=None,
        ):
            self.hpi_values.append(macro_features["hpi"])
            return {"C": 1.0}

    model = RecordingTransitionModel()
    macro_provider = DataFrameMacroFeatureProvider(
        pd.DataFrame(
            {"hpi": [101.0, 102.0]},
            index=pd.to_datetime(["2026-02-28", "2026-03-31"]),
        )
    )
    simulator = LoanSimulator(
        model,
        _cashflow_engine(),
        horizon=2,
        start_date="2026-01-31",
        macro_provider=macro_provider,
    )

    result = simulator.simulate_loan(
        Loan("L1", 1200.0, 0.12, 12, original_balance=1500.0, status="C")
    )

    assert model.hpi_values == [101.0, 102.0]
    assert result.to_frame()["period_date"].tolist() == ["2026-02", "2026-03"]
    assert result.to_frame()["original_balance"].tolist() == [1500.0, 1500.0]


def test_loan_simulator_outputs_recovery_when_lagged_event_is_due():
    """Recovery events created inside the horizon still emit after the horizon."""
    config = _status_config()
    transition_model = ConstantTransitionModel(
        {
            "C": {"LIQ": 1.0},
            "PIF": {"PIF": 1.0},
            "LIQ": {"LIQ": 1.0},
        },
        status_config=config,
    )
    simulator = LoanSimulator(
        transition_model,
        _cashflow_engine(recovery_lag=2),
        horizon=1,
        start_date="2026-01-31",
    )

    frame = simulator.simulate_loan(
        Loan("L1", 1200.0, 0.12, 12, status="C")
    ).to_frame()

    assert frame["period"].tolist() == [1, 3]
    assert frame.loc[frame["period"] == 1, "loss"].iloc[0] == 480.0
    assert frame.loc[frame["period"] == 1, "net_recovery"].iloc[0] == 0.0
    assert frame.loc[frame["period"] == 3, "net_recovery"].iloc[0] == 720.0


def test_portfolio_simulator_outputs_reproducible_aggregates():
    """Stable path seeds make repeated simulations reproducible."""
    config = _status_config()
    transition_model = ConstantTransitionModel(
        {
            "C": {"C": 0.70, "PIF": 0.30},
            "PIF": {"PIF": 1.0},
            "LIQ": {"LIQ": 1.0},
        },
        status_config=config,
    )
    loan = Loan("L1", 1200.0, 0.12, 12, status="C")

    first = PortfolioSimulator(
        LoanSimulator(
            transition_model,
            _cashflow_engine(),
            horizon=3,
            n_paths=3,
            seed=7,
            start_date="2026-01-31",
        )
    ).simulate([loan])
    second = PortfolioSimulator(
        LoanSimulator(
            transition_model,
            _cashflow_engine(),
            horizon=3,
            n_paths=3,
            seed=7,
            start_date="2026-01-31",
        )
    ).simulate([loan])

    pd.testing.assert_frame_equal(
        first.path_cashflows().reset_index(drop=True),
        second.path_cashflows().reset_index(drop=True),
    )
    assert first.loan_cashflows()["loan_id"].unique().tolist() == ["L1"]
    assert set(first.portfolio_cashflows()["period"]).issubset({1, 2, 3})


def test_loan_cashflows_average_over_all_paths_after_early_termination():
    """Early-terminated paths contribute zero to later loan-level averages."""
    loan = Loan("L1", 100.0, 0.0, 12)
    result = LoanSimulationResult(
        loan=loan,
        cashflows=[
            PeriodCashflow(
                loan_id="L1",
                path_id=0,
                period=1,
                begin_age_months=0,
                end_age_months=1,
                begin_balance=100.0,
                end_balance=0.0,
                begin_status="CURRENT",
                end_status="PAID_OFF",
                principal_collected=100.0,
            ),
            PeriodCashflow(
                loan_id="L1",
                path_id=1,
                period=1,
                begin_age_months=0,
                end_age_months=1,
                begin_balance=100.0,
                end_balance=90.0,
                begin_status="CURRENT",
                end_status="CURRENT",
                scheduled_principal=10.0,
                principal_collected=10.0,
            ),
            PeriodCashflow(
                loan_id="L1",
                path_id=1,
                period=2,
                begin_age_months=1,
                end_age_months=2,
                begin_balance=90.0,
                end_balance=80.0,
                begin_status="CURRENT",
                end_status="CURRENT",
                scheduled_principal=10.0,
                principal_collected=10.0,
            ),
        ],
        start_period=pd.Period("2026-01", freq="M"),
        n_paths=2,
    )

    loan_cashflows = PortfolioSimulationResult([result]).loan_cashflows()
    period_two = loan_cashflows.loc[loan_cashflows["period"] == 2].iloc[0]

    assert period_two["begin_balance"] == 45.0
    assert period_two["principal_collected"] == 5.0


def test_loan_simulator_passes_updated_path_features_to_next_period():
    """Path features from period t are visible starting in period t + 1."""
    class RecordingTransitionModel(TransitionModel):
        def __init__(self):
            self.path_features_by_call = []

        def predict(
            self,
            loan,
            current_state,
            macro_features=None,
            path_features=None,
            model_features=None,
        ):
            self.path_features_by_call.append(dict(path_features))
            if current_state.period == 0:
                return {"D1M": 1.0}
            return {"C": 1.0}

    config = StatusConfig(
        valid_statuses={"C", "D1M"},
        terminal_statuses=set(),
        prepay_statuses=set(),
        default_statuses=set(),
        delinquency_buckets={"D1M": "dq30_balance"},
    )
    matrix = {
        "C": {"C": 1, "D1M": 0},
        "D1M": {"C": 2, "D1M": 1},
    }
    model = RecordingTransitionModel()
    simulator = LoanSimulator(
        model,
        CashflowEngine(
            MatrixPaymentPolicy(matrix, status_config=config),
            ConstantSeverityProvider(0.40),
            ConstantRecoveryLagProvider(0),
            status_config=config,
        ),
        horizon=2,
        start_date="2026-01-31",
    )

    simulator.simulate_loan(Loan("L1", 1200.0, 0.12, 12, status="C"))

    assert model.path_features_by_call[0]["ever_delinquent"] is False
    assert model.path_features_by_call[1] == {
        "ever_delinquent": True,
        "months_since_last_delinquency": 0,
        "consecutive_delinquent_months": 1,
        "times_delinquent": 1,
    }


def test_loan_simulator_passes_runtime_model_features_and_updates_provider():
    """Runtime feature providers own per-path model feature lifecycle."""
    class RecordingRuntimeFeatureProvider(RuntimeFeatureProvider):
        def __init__(self):
            self.updates = []

        def initialize_path_state(self, loan, start_period):
            return {"start_period": str(start_period)}

        def model_features_for_period(
            self,
            *,
            loan,
            current_state,
            period_date,
            macro_features,
            path_features,
            feature_state,
        ):
            return {
                "current_period": current_state.period,
                "period_date": str(period_date),
                "start_period": feature_state["start_period"],
            }

        def advance_path_state(self, *, feature_state, cashflow, next_state):
            self.updates.append((cashflow.period, next_state.period))

    class RecordingTransitionModel(TransitionModel):
        def __init__(self):
            self.model_features_by_call = []

        def predict(
            self,
            loan,
            current_state,
            macro_features=None,
            path_features=None,
            model_features=None,
        ):
            self.model_features_by_call.append(dict(model_features))
            return {"C": 1.0}

    provider = RecordingRuntimeFeatureProvider()
    model = RecordingTransitionModel()
    simulator = LoanSimulator(
        model,
        _cashflow_engine(),
        horizon=2,
        start_date="2026-01-31",
        runtime_feature_provider=provider,
    )

    simulator.simulate_loan(Loan("L1", 1200.0, 0.12, 12, status="C"))

    assert model.model_features_by_call == [
        {
            "current_period": 0,
            "period_date": "2026-02",
            "start_period": "2026-01",
        },
        {
            "current_period": 1,
            "period_date": "2026-03",
            "start_period": "2026-01",
        },
    ]
    assert provider.updates == [(1, 1), (2, 2)]
