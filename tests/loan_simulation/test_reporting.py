import pandas as pd

from quantbullet.loan_simulation import (
    Loan,
    LoanSimulationResult,
    PeriodCashflow,
    PortfolioSimulationResult,
    simulation_result_frames,
    write_simulation_workbook,
)


def _portfolio_result():
    loan = Loan("L1", balance=1_000.0, annual_rate=0.12, term_months=12)
    cashflow = PeriodCashflow(
        loan_id="L1",
        path_id=0,
        period=1,
        begin_age_months=0,
        end_age_months=1,
        begin_balance=1_000.0,
        end_balance=900.0,
        begin_status="CURRENT",
        end_status="CURRENT",
        scheduled_interest=10.0,
        scheduled_principal=100.0,
        interest_collected=10.0,
        principal_collected=100.0,
    )
    return PortfolioSimulationResult(
        [
            LoanSimulationResult(
                loan=loan,
                cashflows=[cashflow],
                start_period=pd.Period("2026-01", freq="M"),
            )
        ]
    )


def test_simulation_result_frames_include_standard_outputs():
    frames = simulation_result_frames(_portfolio_result())

    assert set(frames) == {
        "portfolio_cashflows",
        "portfolio_metrics",
        "loan_cashflows",
        "path_cashflows",
    }
    assert frames["portfolio_cashflows"]["period"].tolist() == [1]
    assert frames["portfolio_metrics"]["period"].tolist() == [1]


def test_write_simulation_workbook_writes_standard_and_extra_sheets(tmp_path):
    output_path = tmp_path / "simulation.xlsx"
    frames = write_simulation_workbook(
        _portfolio_result(),
        output_path,
        include_loan_cashflows=False,
        extra_sheets={"run_config": pd.DataFrame([{"key": "seed", "value": 1}])},
    )

    assert output_path.exists()
    assert "loan_cashflows" not in frames
    workbook = pd.ExcelFile(output_path)
    assert set(workbook.sheet_names) == {
        "run_config",
        "portfolio_cashflows",
        "portfolio_metrics",
        "path_cashflows",
    }
