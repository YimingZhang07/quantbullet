"""Additive design matrices on fake mortgage data: one runnable example per test method.

These examples use ``quantbullet.preprocessing.Design`` the way a user would: describe the columns and build the
matrix. ``setUpClass`` runs once and holds only the fake loan-months every example starts from
(``quantbullet.utils.data.generate_fake_loan_months``). Each test method builds its own design.

Run one example with the run button next to its method in the VS Code Test panel (``examples`` must be listed
in ``python.testing.pytestArgs``); ``setUpClass`` runs first, then that method. From the repository root:

    .venv/Scripts/python.exe -m pytest examples/preprocessing/test_design_examples.py -k categorical

Every example prints its first rows and writes them to its own sheet of one workbook in OUTPUT_DIR, which git
ignores: the raw column next to the matrix it becomes. The cross examples print only the cross's columns; their
sheets hold the whole matrix. Running an example adds its sheet, or replaces it, and leaves the other sheets as they
are. Close the workbook in Excel before running, since Excel locks an open file.
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quantbullet.preprocessing import Cross, Design, OneHot, Tent
from quantbullet.reporting import ColumnFormat, ColumnMeta, ColumnSchema, ExcelExporter
from quantbullet.utils.data import generate_fake_loan_months

# Where the examples write their workbook. Change it to write somewhere else.
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "design"
WORKBOOK = "design_examples.xlsx"
FIRST_ROWS = 30                                  # rows of each matrix to print and write
INCENTIVE_RANGE = (-3.0, 3.0)                    # percentage points: loans in and out of the money
INCENTIVE_KNOTS = (-3, -2, -1, 0, 1, 2, 3)
BALANCE_KNOTS = (50_000, 250_000, 1_000_000)     # dollars; the fake data's balance x incentive effect is 0 at 250K


def first_rows(raw: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    """The first rows of the raw columns next to the design columns they become."""
    return pd.concat([raw.head(FIRST_ROWS), design.head(FIRST_ROWS)], axis=1).rename_axis("row")


def show(title: str, table: pd.DataFrame) -> None:
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(f"\n== {title}\n{table}")


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def write_sheet(sheet: str, table: pd.DataFrame, decimals: int) -> Path:
    """Write ``table`` to ``sheet`` of the workbook, numbers with ``decimals`` places (row numbers with none).

    ``ExcelExporter`` fits each column's width to its values and header, and adds the sheet to an existing
    workbook, or replaces it, keeping the other sheets.
    """
    path = output_path(WORKBOOK)
    rows = table.reset_index()
    schema = ColumnSchema([ColumnMeta(name=column, display_name=column,
                                      format=ColumnFormat(decimals=0 if column == "row" else decimals))
                           for column in rows.columns])
    ExcelExporter(str(path)).add_sheet(sheet, rows, schema=schema).save()
    print(f"\nwritten: {path} [{sheet}]")
    return path


class TestDesignExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Every example starts from the same fake loan-months.
        cls.loans = generate_fake_loan_months(n_rows=100_000, seed=0, incentive_range=INCENTIVE_RANGE)

    def test_categorical(self):
        """One categorical column: one-hot with purchase as the reference, plus an intercept."""
        design = Design(
            [OneHot("purpose",
                    reference="P")],   # P's column is dropped, so C and N read relative to purchase;
                                       # reference=None keeps all three (then intercept=False)
            intercept=True,            # the first column is all ones
        )
        Z = design.fit_transform(self.loans)   # learns the levels P, C, N from the data
        print(design.groups())                  # {term: its columns in Z}
        table = first_rows(self.loans[["purpose"]], Z)
        show("purpose and its columns", table)
        path = write_sheet("categorical", table, decimals=0)

        self.assertEqual(list(Z.columns), ["intercept", "purpose_C", "purpose_N"])
        self.assertTrue(np.all(Z["purpose_C"] + Z["purpose_N"] == (self.loans["purpose"] != "P")))
        self.assertIn("categorical", pd.ExcelFile(path).sheet_names)

    def test_continuous(self):
        """One continuous column: a tent basis with incentive 0 as the reference, plus an intercept."""
        design = Design(
            [Tent("incentive",
                  knots=INCENTIVE_KNOTS,   # one column per knot: weights interpolate linearly between knots
                  reference=0)],           # 0's column is dropped, so the curve is read relative to incentive 0
            intercept=True,
        )
        Z = design.fit_transform(self.loans)
        print(design.groups())
        table = first_rows(self.loans[["incentive"]], Z).sort_values("incentive")   # low to high incentive
        show("incentive and its columns", table.round(4))
        path = write_sheet("continuous", table, decimals=4)

        self.assertEqual(list(Z.columns), ["intercept", "incentive_at_-3", "incentive_at_-2", "incentive_at_-1",
                                           "incentive_at_1", "incentive_at_2", "incentive_at_3"])
        self.assertTrue(np.all(Z.drop(columns="intercept").sum(axis=1) <= 1 + 1e-12))  # 0's weight is the rest
        self.assertIn("continuous", pd.ExcelFile(path).sheet_names)

    def test_categorical_cross(self):
        """Two categorical columns and their cross: each purpose's effect may differ by year."""
        purpose = OneHot("purpose", reference="P")
        year = OneHot("year", reference=2019)          # integer levels work too: year_2020, ..., year_2025
        design = Design(
            [purpose, year,
             Cross(purpose, year,                      # one column per (purpose, year) cell, e.g. purpose_C_x_year_2020;
                                                       # the same objects share their levels and references
                   drop=["purpose", "year"])],         # drop the cells at P or at 2019: the main terms and the
                                                       # intercept already span them (Design.check names the drop needed)
            intercept=True,
        )
        Z = design.fit_transform(self.loans)
        print(design.groups())
        cross = design.groups()["purpose_x_year"]
        table = first_rows(self.loans[["purpose", "year"]], Z).sort_values(["purpose", "year"])
        show("purpose, year and their cross columns", table[["purpose", "year", *cross]])
        path = write_sheet("categorical cross", table, decimals=0)

        self.assertEqual(len(cross), 2 * 6)                                        # (C, N) x (2020, ..., 2025)
        off_reference = (self.loans["purpose"] != "P") & (self.loans["year"] != 2019)
        self.assertTrue(np.all(Z[cross].sum(axis=1) == off_reference))            # one cell is 1, off the references
        self.assertIn("categorical cross", pd.ExcelFile(path).sheet_names)

    def test_continuous_cross(self):
        """Two continuous columns and their cross: the incentive curve may bend with balance, as in the fake data."""
        balance = Tent("balance", knots=BALANCE_KNOTS, reference=250_000)
        incentive = Tent("incentive", knots=INCENTIVE_KNOTS, reference=0)
        design = Design(
            [balance, incentive,
             Cross(balance, incentive,                 # one column per (balance knot, incentive knot), the product of
                                                       # their tent weights
                   drop=["balance", "incentive"])],    # drop the cells at 250K or at incentive 0, as above
            intercept=True,
        )
        Z = design.fit_transform(self.loans)
        print(design.groups())
        cross = design.groups()["balance_x_incentive"]
        table = first_rows(self.loans[["balance", "incentive"]], Z).sort_values("balance")   # small to large loans
        show("balance, incentive and their cross columns", table[["balance", "incentive", *cross]].round(4))
        path = write_sheet("continuous cross", table, decimals=4)

        self.assertEqual(len(cross), 2 * 6)                                        # (50K, 1M) x 6 incentive knots
        corner = "balance_at_1000000_x_incentive_at_3"
        self.assertTrue(np.allclose(Z[corner], Z["balance_at_1000000"] * Z["incentive_at_3"]))
        self.assertIn("continuous cross", pd.ExcelFile(path).sheet_names)
