import pandas as pd

from quantbullet.loan_simulation import (
    CashflowEngine,
    ConstantRecoveryLagProvider,
    ConstantSeverityProvider,
    ConstantTransitionModel,
    DataFrameMacroFeatureProvider,
    Loan,
    LoanSimulator,
    MatrixPaymentPolicy,
    PortfolioSimulator,
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
    class RecordingTransitionModel(TransitionModel):
        def __init__(self):
            self.hpi_values = []

        def predict(
            self,
            loan,
            current_state,
            macro_features=None,
            path_features=None,
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

    result = simulator.simulate_loan(Loan("L1", 1200.0, 0.12, 12, status="C"))

    assert model.hpi_values == [101.0, 102.0]
    assert result.to_frame()["period_date"].tolist() == ["2026-02", "2026-03"]


def test_loan_simulator_outputs_recovery_when_lagged_event_is_due():
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
