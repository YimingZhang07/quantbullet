from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .metrics import compute_period_metrics
from .simulator import PortfolioSimulationResult


def simulation_result_frames(
    result: PortfolioSimulationResult,
    *,
    include_loan_cashflows: bool = True,
    include_path_cashflows: bool = True,
    include_portfolio_metrics: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build standard DataFrame outputs for a portfolio simulation result."""
    portfolio_cashflows = result.portfolio_cashflows()
    frames = {"portfolio_cashflows": portfolio_cashflows}

    if include_portfolio_metrics:
        frames["portfolio_metrics"] = compute_period_metrics(portfolio_cashflows)
    if include_loan_cashflows:
        frames["loan_cashflows"] = result.loan_cashflows()
    if include_path_cashflows:
        frames["path_cashflows"] = result.path_cashflows()

    return frames


def write_simulation_workbook(
    result: PortfolioSimulationResult,
    output_path: str | Path,
    *,
    include_loan_cashflows: bool = True,
    include_path_cashflows: bool = True,
    include_portfolio_metrics: bool = True,
    extra_sheets: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Write standard simulation outputs to an Excel workbook.

    Returns the frames that were written so callers can reuse them for logging
    or quick console summaries without recomputing.
    """
    frames = simulation_result_frames(
        result,
        include_loan_cashflows=include_loan_cashflows,
        include_path_cashflows=include_path_cashflows,
        include_portfolio_metrics=include_portfolio_metrics,
    )
    output_path = Path(output_path)
    with pd.ExcelWriter(output_path) as writer:
        if extra_sheets is not None:
            for sheet_name, frame in extra_sheets.items():
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return frames
