"""Two-way diagnostics on fake loan-month data: one runnable example per test method.

These examples use ``quantbullet.diagnostics.TwoWayDiagnostic`` the way a user would. ``setUpClass`` runs once
and holds only what every example starts from: the fake loans and the column settings for
``TwoWayDiagnostic``. Each test method builds its own diagnostic and shows one usage.

Run one example with the run button next to its method in the VS Code Test panel (``examples`` must be listed
in ``python.testing.pytestArgs``); ``setUpClass`` runs first, then that method. From the repository root:

    .venv/Scripts/python.exe -m pytest examples/diagnostics/test_two_way_examples.py -k ae_table

Tables print to the console. Figures, CSVs and PDFs are written to OUTPUT_DIR, which git ignores; open them
from there.

The fake data has a known effect that the "model" misses: larger loans respond more strongly to incentive.
Its true multiplier on the model is exp(0.06 * log(balance / 250K) * (incentive + 3)). So actual / model rises
with incentive for large loans and falls for small ones, and it is exactly 1 on the 250K row and the -3 column.
"""
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files; no plot windows

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from quantbullet.diagnostics import TwoWayDiagnostic  # noqa: E402
from quantbullet.plot import PRINT_THEME, BinSpec  # noqa: E402

# Where the examples write their files. Change it to write somewhere else.
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "two_way"

# Bins for the A/E examples: right-closed intervals, the lowest edge included.
BALANCE_BINS = BinSpec.edges([30_000, 100_000, 200_000, 350_000, 600_000, 1_000_000])
INCENTIVE_BINS = BinSpec.edges([-5, -4, -3, -2, -1, 0])
# Knots for the correction examples: D is bilinear between knots and flat beyond the end knots.
BALANCE_KNOTS = (50_000, 150_000, 250_000, 450_000, 800_000)
INCENTIVE_KNOTS = (-5, -4, -3, -2, -1, 0)
# Fixed scales for figures that are compared side by side, e.g. before and after a correction: colours saturate
# at x and / color_range, and the slices share y_range. Otherwise each figure fits its own data.
AE_COLOR_RANGE, AE_Y_RANGE = 1.3, (0.7, 1.35)
D_COLOR_RANGE, D_Y_RANGE = 1.45, (0.6, 1.55)


def fake_loan_months(n_rows: int, seed: int) -> pl.DataFrame:
    """Fake loan-month rows, and model predictions that miss the balance x incentive interaction.

    Columns: ``balance``, ``incentive`` (percentage points), ``year``, ``prepaid`` (1 if the loan paid off that
    month), ``model_smm`` (the model's predicted monthly prepay rate), ``model_without_balance`` (a worse model
    that also misses the balance effect) and ``upb`` (the weight, equal to balance).
    """
    rng = np.random.default_rng(seed)
    balance = np.exp(rng.uniform(np.log(30_000), np.log(1_000_000), n_rows))
    incentive = rng.uniform(-5.0, 0.0, n_rows)
    year = rng.integers(2019, 2026, n_rows)
    size = np.log(balance / 250_000)
    # The model: one incentive slope for every balance, plus a small balance effect.
    model_without_balance = 0.01 * np.exp(0.25 * (incentive + 3))
    model_smm = model_without_balance * np.exp(-0.05 * size)
    # The truth: the incentive slope grows with balance. This is what the diagnostics should find.
    true_smm = model_smm * np.exp(0.06 * size * (incentive + 3))
    prepaid = (rng.random(n_rows) < true_smm).astype(float)
    return pl.DataFrame({"balance": balance, "incentive": incentive, "year": year, "prepaid": prepaid,
                         "model_smm": model_smm, "model_without_balance": model_without_balance, "upb": balance})


def show(title: str, table: pd.DataFrame) -> None:
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(f"\n== {title}\n{table}")


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


class TestTwoWayExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Every example starts from the same fake loans ...
        cls.loans = fake_loan_months(n_rows=2_000_000, seed=0)
        # ... and describes them to TwoWayDiagnostic the same way.
        cls.columns = dict(
            x="balance",          # heatmap rows; one slice panel per x bin or knot
            z="incentive",        # heatmap columns; the horizontal axis of each slice
            y="prepaid",          # what happened
            pred="model_smm",     # what the model predicted, row by row
            weight="upb",         # weight rows by balance; None weights every row equally
            x_scale="log",        # balance is drawn, and interpolated between knots, on a log scale
            z_scale="linear",
            labels={"balance": "current balance", "incentive": "incentive (pp)"},
        )

    # ---------------------------------------------------------------- binned A/E
    def test_ae_table(self):
        """Binned actual / expected: one value per (balance bin, incentive bin) cell."""
        diag = TwoWayDiagnostic(self.loans, **self.columns)
        print(diag)  # rows kept, and rows excluded by reason
        ae = diag.ae(x_bins=BALANCE_BINS, z_bins=INCENTIVE_BINS)
        show("A/E: balance bins down, incentive bins across", ae.pivot("ae").round(2))
        show("standard error of A/E", ae.pivot("ae_se").round(3))
        show("rows per cell", ae.pivot("rows"))
        path = output_path("ae_table.csv")
        ae.table.to_csv(path, index=False)  # actual, expected, ae, ae_se and the means of every cell
        print(f"\nwritten: {path}")

        small_loans, large_loans = ae.pivot("ae").iloc[0], ae.pivot("ae").iloc[-1]
        self.assertGreater(small_loans.iloc[0], small_loans.iloc[-1])   # falls with incentive
        self.assertLess(large_loans.iloc[0], large_loans.iloc[-1])      # rises with incentive

    def test_ae_plot(self):
        """Heatmap of A/E above one slice per balance bin."""
        diag = TwoWayDiagnostic(self.loans, **self.columns)
        ae = diag.ae(x_bins=BALANCE_BINS, z_bins=INCENTIVE_BINS)
        plot = ae.plot(
            slices="x",               # one panel per balance bin; "z" gives one per incentive bin
            min_rows=200,             # cells with fewer rows are left blank
            ci=0.95,                  # error bars on A/E
            n_cols=3,                 # slice panels per row
            panel_size=(3.4, 2.6),    # inches per slice panel
            heatmap_height=3.2,       # inches
            title="Fake loans: actual / model by balance and incentive",
        )
        path = output_path("ae.png")
        plot.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nwritten: {path}")
        self.assertEqual(len(plot.slice_axes), 5)

    def test_ae_plot_by_incentive(self):
        """The same A/E with one panel per incentive bin: balance runs along each slice, on a log axis."""
        diag = TwoWayDiagnostic(self.loans, **self.columns)
        ae = diag.ae(x_bins=BALANCE_BINS, z_bins=INCENTIVE_BINS)
        plot = ae.plot(slices="z", min_rows=200, ci=0.95, n_cols=5, panel_size=(3.0, 2.6), heatmap_height=3.2,
                       title="Fake loans: actual / model, one panel per incentive bin")
        path = output_path("ae_by_incentive.png")
        plot.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nwritten: {path}")
        self.assertEqual(len(plot.slice_axes), 5)

    # ---------------------------------------------------------------- correction surface D
    def test_correction_interaction(self):
        """Fit D(balance, incentive) with both references: D holds only the interaction the model misses."""
        diag = TwoWayDiagnostic(self.loans, **self.columns)
        fit = diag.correction(
            x_knots=BALANCE_KNOTS,
            z_knots=INCENTIVE_KNOTS,
            x_reference=250_000,  # D = 1 along this row ...
            z_reference=-3,       # ... and this column, so D cannot redo the model's main effects
            loss="poisson",       # y is a count (0/1 here); use "mse" for a continuous y
            ridge=0.01,           # shrinks nodes with little data toward 1
            max_iter=500,         # L-BFGS-B iteration cap ...
            tol=1e-14,            # ... and its ftol; both are the defaults
        )
        show("D at the knots (1 = the model is right)", fit.table("nodes").round(3))
        show("standard errors", fit.table("se").round(3))
        show("(D - 1) / se", fit.table("z_score").round(1))
        show("rows behind each node (interpolation-weighted)", fit.table("rows").round(0))
        print(f"\ndeviance: {fit.deviance}\niterations={fit.iterations} converged={fit.converged} "
              f"clamped beyond the end knots={fit.clamped}")

        truth = np.exp(0.06 * np.log(fit.x_knots / 250_000)[:, None] * (fit.z_knots + 3)[None, :])
        show("the true multiplier", pd.DataFrame(truth, index=fit.x_knots, columns=fit.z_knots).round(3))
        self.assertTrue(fit.converged)
        self.assertTrue(np.all(np.abs(fit.nodes - truth) < 4 * fit.se + 1e-12))  # within 4 SE of the truth

    def test_correction_plot(self):
        """Heatmap of D above one slice per balance knot."""
        diag = TwoWayDiagnostic(self.loans, **self.columns)
        fit = diag.correction(x_knots=BALANCE_KNOTS, z_knots=INCENTIVE_KNOTS, x_reference=250_000,
                              z_reference=-3, loss="poisson", ridge=0.01)
        plot = fit.plot(
            slices="x",               # one panel per balance knot; the reference row is all ones and is skipped
            ci=0.95,
            label="fitted D",         # legend entry for this fit
            n_cols=4,
            panel_size=(3.2, 2.6),
            heatmap_height=3.4,
            title="Fake loans: interaction the model misses",
        )
        path = output_path("correction_interaction.png")
        plot.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nwritten: {path}")
        self.assertEqual(len(plot.slice_axes), len(BALANCE_KNOTS) - 1)

    def test_correction_full_surface(self):
        """No references: D shows every way a model is off, here a missed balance effect plus the interaction."""
        diag = TwoWayDiagnostic(self.loans, **{**self.columns, "pred": "model_without_balance"})
        fit = diag.correction(
            x_knots=BALANCE_KNOTS,
            z_knots=INCENTIVE_KNOTS,
            x_reference=None,     # nothing is fixed at 1
            z_reference=None,
            loss="poisson",
            ridge=0.01,
        )
        show("D without references", fit.table("nodes").round(3))
        plot = fit.plot(slices="x", ci=0.95, label="fitted D", n_cols=5, panel_size=(3.0, 2.6), heatmap_height=3.4,
                        title="Fake loans: full surface against a model without a balance effect")
        path = output_path("correction_full_surface.png")
        plot.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nwritten: {path}")
        self.assertTrue(fit.free.all())

    def test_correction_by_period(self):
        """Fit the same surface on two periods and overlay them on the full-sample fit."""
        fits = {}
        for name, loans in (("all years", self.loans),
                            ("2019-2021", self.loans.filter(pl.col("year") <= 2021)),
                            ("2022-2025", self.loans.filter(pl.col("year") >= 2022))):
            diag = TwoWayDiagnostic(loans, **self.columns)
            fits[name] = diag.correction(x_knots=BALANCE_KNOTS, z_knots=INCENTIVE_KNOTS, x_reference=250_000,
                                         z_reference=-3, loss="poisson", ridge=0.01)
            show(f"D, {name}", fits[name].table("nodes").round(3))
        plot = fits["all years"].plot(
            overlays={"2019-2021": fits["2019-2021"], "2022-2025": fits["2022-2025"]},  # drawn as lines
            slices="x", ci=0.95, label="all years", n_cols=4, panel_size=(3.2, 2.6), heatmap_height=3.4,
            title="Fake loans: is the interaction stable over time?",
        )
        path = output_path("correction_by_period.png")
        plot.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nwritten: {path}")
        self.assertEqual(len(plot.fig.legends[0].get_texts()), 3)

    def test_apply_the_correction(self):
        """Multiply the model by D and check A/E again: the tilt should be gone (in sample)."""
        diag = TwoWayDiagnostic(self.loans, **self.columns)
        fit = diag.correction(x_knots=BALANCE_KNOTS, z_knots=INCENTIVE_KNOTS, x_reference=250_000,
                              z_reference=-3, loss="poisson", ridge=0.01)
        corrected = self.loans.with_columns(
            pl.Series("corrected_smm", self.loans["model_smm"].to_numpy() * fit.predict(self.loans)))

        before = diag.ae(x_bins=BALANCE_BINS, z_bins=INCENTIVE_BINS)
        after = TwoWayDiagnostic(corrected, **{**self.columns, "pred": "corrected_smm"}).ae(
            x_bins=BALANCE_BINS, z_bins=INCENTIVE_BINS)
        show("A/E before the correction", before.pivot("ae").round(2))
        show("A/E after the correction", after.pivot("ae").round(2))
        spread_before = float(np.nanstd(before.pivot("ae").to_numpy()))
        spread_after = float(np.nanstd(after.pivot("ae").to_numpy()))
        print(f"\nspread of A/E across cells: before {spread_before:.3f}, after {spread_after:.3f}")
        self.assertLess(spread_after, spread_before)

    # ---------------------------------------------------------------- output
    def test_before_and_after_report(self):
        """Four PDF pages: A/E and D for the model, then for the model corrected by D (in sample)."""
        # Before: diagnose the model, and fit the correction it needs.
        before = TwoWayDiagnostic(self.loans, **self.columns)
        before_ae = before.ae(x_bins=BALANCE_BINS, z_bins=INCENTIVE_BINS)
        before_fit = before.correction(x_knots=BALANCE_KNOTS, z_knots=INCENTIVE_KNOTS, x_reference=250_000,
                                       z_reference=-3, loss="poisson", ridge=0.01)
        # After: multiply the model by D, then run the same two diagnostics on the corrected prediction.
        corrected = self.loans.with_columns(
            pl.Series("corrected_smm", self.loans["model_smm"].to_numpy() * before_fit.predict(self.loans)))
        after = TwoWayDiagnostic(corrected, **{**self.columns, "pred": "corrected_smm"})
        after_ae = after.ae(x_bins=BALANCE_BINS, z_bins=INCENTIVE_BINS)
        after_fit = after.correction(x_knots=BALANCE_KNOTS, z_knots=INCENTIVE_KNOTS, x_reference=250_000,
                                     z_reference=-3, loss="poisson", ridge=0.01)

        def spread(ae):  # how far A/E strays from 1 across cells
            return float(np.nanstd(ae.pivot("ae").to_numpy()))

        pages = [
            (before_ae, AE_COLOR_RANGE, AE_Y_RANGE,
             f"1. Before: actual / model\nA/E spread across cells {spread(before_ae):.3f}"),
            (before_fit, D_COLOR_RANGE, D_Y_RANGE,
             f"2. Before: the correction D the model needs\n"
             f"deviance change from applying D {before_fit.deviance['change']:+.3%}"),
            (after_ae, AE_COLOR_RANGE, AE_Y_RANGE,
             f"3. After: actual / corrected model (model x D)\nA/E spread across cells {spread(after_ae):.3f}"),
            (after_fit, D_COLOR_RANGE, D_Y_RANGE,
             f"4. After: D fitted again on the corrected model\n"
             f"deviance change from applying it {after_fit.deviance['change']:+.3%}"),
        ]
        path = output_path("before_and_after.pdf")
        with PdfPages(path) as pdf:
            for result, color_range, y_range, title in pages:
                plot = result.plot(
                    slices="x", ci=0.95, n_cols=3, panel_size=(2.6, 1.9), heatmap_height=2.8,
                    color_range=color_range,   # the same scales on the before and after pages,
                    y_range=y_range,           # so colours and heights compare directly
                    title=title, theme=PRINT_THEME,   # PRINT_THEME: font sizes for print
                )
                pdf.savefig(plot.fig)
        print(f"\nA/E spread across cells: before {spread(before_ae):.3f}, after {spread(after_ae):.3f}")
        print(f"largest |D - 1|: before {np.abs(before_fit.nodes - 1).max():.3f}, "
              f"after {np.abs(after_fit.nodes - 1).max():.3f}")
        print(f"written: {path}")

        self.assertLess(spread(after_ae), spread(before_ae))
        self.assertGreater(np.abs(before_fit.nodes - 1).max(), 0.2)   # the model misses the interaction ...
        self.assertLess(np.abs(after_fit.nodes - 1).max(), 0.05)      # ... and the corrected model does not
