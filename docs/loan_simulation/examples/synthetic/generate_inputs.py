"""Generate deterministic CSV inputs for the synthetic loan simulation example."""

from __future__ import annotations

import calendar
import csv
import math
import random
from collections.abc import Iterable, Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from quantbullet.loan_simulation import LoanStatus


EXAMPLE_DIR = Path(__file__).resolve().parent
INPUT_DIR = EXAMPLE_DIR / "input"

N_LOANS = 100
ORIGINATION_DATE = date(2000, 1, 31)
MACRO_START_DATE = date(1999, 6, 1)
MACRO_END_DATE = date(2010, 12, 31)

LOAN_SEED = 20000131
HPI_SEED = 19990601
RATE_SEED = 20101231

LOAN_TERM_MONTHS = (36, 60, 120)
MIN_ORIGINATION_SPREAD = 0.01
MAX_ORIGINATION_SPREAD = 0.04


def month_ends(start_date: date, end_date: date) -> Iterator[date]:
    """Yield month-end dates for every month covered by the inclusive range."""
    year = start_date.year
    month = start_date.month

    while (year, month) <= (end_date.year, end_date.month):
        period_date = date(year, month, calendar.monthrange(year, month)[1])
        if start_date <= period_date <= end_date:
            yield period_date

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def build_loan_rows(origination_base_rate: float) -> list[dict[str, Any]]:
    """Build 100 deterministic, newly originated current loans."""
    rng = random.Random(LOAN_SEED)
    rows = []

    for loan_number in range(1, N_LOANS + 1):
        original_balance = rng.randrange(500, 5_001) * 100
        annual_rate = origination_base_rate + rng.uniform(
            MIN_ORIGINATION_SPREAD,
            MAX_ORIGINATION_SPREAD,
        )
        rows.append(
            {
                "loan_id": f"SYN{loan_number:04d}",
                "origination_date": ORIGINATION_DATE.isoformat(),
                "original_balance": f"{original_balance:.2f}",
                "balance": f"{original_balance:.2f}",
                "annual_rate": f"{annual_rate:.6f}",
                "term_months": rng.choice(LOAN_TERM_MONTHS),
                "age_months": 0,
                "status": LoanStatus.CURRENT,
            }
        )

    return rows


def build_hpi_rows(period_dates: list[date]) -> list[dict[str, Any]]:
    """Build a seeded monthly HPI random walk starting at 100."""
    rng = random.Random(HPI_SEED)
    hpi = 100.0
    rows = []

    for index, period_date in enumerate(period_dates):
        if index:
            monthly_growth = rng.gauss(0.002, 0.004)
            hpi = max(hpi * (1.0 + monthly_growth), 1.0)
        rows.append({"date": period_date.isoformat(), "hpi": f"{hpi:.4f}"})

    return rows


def build_rate_rows(period_dates: list[date]) -> list[dict[str, Any]]:
    """Build a seeded, mean-reverting monthly market-rate series."""
    rng = random.Random(RATE_SEED)
    market_rate = 0.075
    rows = []

    for index, period_date in enumerate(period_dates):
        if index:
            target_rate = 0.065 + 0.01 * math.sin(index / 12.0)
            market_rate += 0.15 * (target_rate - market_rate)
            market_rate += rng.gauss(0.0, 0.0015)
            market_rate = min(max(market_rate, 0.02), 0.12)
        rows.append(
            {
                "date": period_date.isoformat(),
                "market_rate": f"{market_rate:.6f}",
            }
        )

    return rows


def write_csv(
    path: Path,
    *,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    period_dates = list(month_ends(MACRO_START_DATE, MACRO_END_DATE))
    hpi_rows = build_hpi_rows(period_dates)
    rate_rows = build_rate_rows(period_dates)
    origination_date = ORIGINATION_DATE.isoformat()
    origination_base_rate = next(
        float(row["market_rate"])
        for row in rate_rows
        if row["date"] == origination_date
    )
    loan_rows = build_loan_rows(origination_base_rate)

    write_csv(
        INPUT_DIR / "loans.csv",
        fieldnames=[
            "loan_id",
            "origination_date",
            "original_balance",
            "balance",
            "annual_rate",
            "term_months",
            "age_months",
            "status",
        ],
        rows=loan_rows,
    )
    write_csv(
        INPUT_DIR / "hpi.csv",
        fieldnames=["date", "hpi"],
        rows=hpi_rows,
    )
    write_csv(
        INPUT_DIR / "rates.csv",
        fieldnames=["date", "market_rate"],
        rows=rate_rows,
    )

    print(f"Wrote {len(loan_rows)} loans to {INPUT_DIR / 'loans.csv'}")
    print(f"Wrote {len(hpi_rows)} HPI rows to {INPUT_DIR / 'hpi.csv'}")
    print(f"Wrote {len(rate_rows)} rate rows to {INPUT_DIR / 'rates.csv'}")


if __name__ == "__main__":
    main()
