import importlib.util
import sys
from pathlib import Path

import pandas as pd


SYNTHETIC_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "loan_simulation"
    / "examples"
    / "synthetic"
)
sys.path.insert(0, str(SYNTHETIC_DIR))

RUN_SPEC = importlib.util.spec_from_file_location(
    "synthetic_loan_simulation_run",
    SYNTHETIC_DIR / "run.py",
)
assert RUN_SPEC is not None
assert RUN_SPEC.loader is not None
synthetic_run = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(synthetic_run)


def test_synthetic_inputs_load_into_engine_types():
    loans, start_period = synthetic_run.load_loans()
    macro = synthetic_run.load_macro_features()

    assert len(loans) == 100
    assert start_period == pd.Period("2000-01", freq="M")
    assert {loan.term_months for loan in loans} == {36, 60, 120}
    assert set(macro.columns) == {"hpi", "market_rate"}
    assert macro.index.min() == pd.Timestamp("1999-06-30")
    assert macro.index.max() == pd.Timestamp("2010-12-31")


def test_synthetic_run_writes_standard_workbook(tmp_path, monkeypatch):
    output_path = tmp_path / "synthetic_smoke.xlsx"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--max-loans",
            "2",
            "--n-paths",
            "2",
            "--horizon",
            "3",
            "--output",
            str(output_path),
        ],
    )

    synthetic_run.main()

    workbook = pd.ExcelFile(output_path)
    assert set(workbook.sheet_names) == {
        "portfolio_cashflows",
        "portfolio_metrics",
        "loan_cashflows",
    }
    portfolio = pd.read_excel(workbook, sheet_name="portfolio_cashflows")
    assert portfolio["period"].tolist() == [1, 2, 3]
