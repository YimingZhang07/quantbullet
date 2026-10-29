"""Design reconciled with another implementation of additive design columns: one runnable example per test method.

The other implementation builds its columns from specs. A spec is a sequence of (variable, knots or levels, reference)
triples: a continuous variable becomes one tent column per knot, a categorical one an indicator per listed level, and
the factors of one spec are multiplied cell by cell. A cell is dropped when any factor sits on one of its references;
the reference is a 1-based index into the knots or levels, a list of them, or None to keep every piece. The
``reference_*`` functions below are a port of it. Each example builds one of its specs and the ``Design`` that holds
the same cells, and compares them cell by cell: whether each side keeps the cell, and how far apart the values are.

The values agree to rounding: within TOLERANCE, a unit in the last place, and exactly where knots are 1 apart.
``Design`` computes the smaller of a value's two tent weights as 1 minus the larger, so its rows sum to exactly 1; the
other implementation divides for each weight. Where the designs differ, the examples show why. The other
implementation
- has no intercept, since an offset carries the level, and
- writes a cross without its main terms or lower crosses;
  ``Design`` builds both as written and lists each constraint they add as a ``Violation`` on ``violations_``, with one
  ``DesignWarning``;
- may cross a single level (no-cash-out refi only), and
- may give one factor two references (UPB at 250K and 400K);
  ``Design`` writes both with a list of references, ``reference=["C", "P"]`` for a cross of N alone, and records the
  constraint each further reference adds.
So every spec is written exactly, and its cells match.

``setUpClass`` runs once and holds only the fake loan-months every example starts from
(``quantbullet.utils.data.generate_fake_loan_months``). Run one example with the run button next to its method in the
VS Code Test panel (``examples`` must be listed in ``python.testing.pytestArgs``). From the repository root:

    .venv/Scripts/python.exe -m pytest examples/preprocessing/test_design_reconciliation_examples.py -k cont_x_cont

Every example prints a table with one row per cell of its spec and writes it to its own sheet of one workbook in
OUTPUT_DIR, which git ignores. Close the workbook in Excel before running, since Excel locks an open file.

The other implementation codes purposes p, rt and co; here they are P (purchase), N (no-cash-out or rate/term refi)
and C (cash-out refi).
"""
import itertools
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quantbullet.preprocessing import Cross, Design, DesignWarning, OneHot, Tent
from quantbullet.reporting import ColumnFormat, ColumnMeta, ColumnSchema, ExcelExporter
from quantbullet.utils.data import generate_fake_loan_months

# Where the examples write their workbook. Change it to write somewhere else.
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "design"
WORKBOOK = "design_reconciliation.xlsx"
INCENTIVE_RANGE = (-3.0, 3.0)                    # percentage points, so loans fall below and above every knot
INCENTIVE_KNOTS = [-1.0, 0.0, 1.0, 2.0]
FICO_KNOTS = [680.0, 720.0, 760.0, 800.0]
UPB_KNOTS = [150_000.0, 250_000.0, 400_000.0, 700_000.0]
PURPOSES = ["P", "N", "C"]
TOLERANCE = 1e-15                                # values that differ by rounding only


def reference_tent_basis(values, knots) -> np.ndarray:
    """The other implementation's tents: column i is 1 at knots[i] and splits linearly with its neighbours; below the
    first knot the first column is 1, from the last knot on the last column is 1."""
    values = np.asarray(values, dtype=float)
    knots = np.asarray(knots, dtype=float)
    out = np.zeros((len(values), len(knots)))
    out[values < knots[0], 0] = 1.0
    out[values >= knots[-1], -1] = 1.0
    for i in range(len(knots) - 1):
        left, right = knots[i], knots[i + 1]
        inside = (values >= left) & (values < right)
        out[inside, i] = (right - values[inside]) / (right - left)
        out[inside, i + 1] = (values[inside] - left) / (right - left)
    return out


def reference_cells(frame: pd.DataFrame, spec: tuple):
    """Every product cell of one spec, as the other implementation builds them.

    Returns the cells' values ``{name: column}`` (names such as ``_purposeN_incentive3``, knots by 1-based index), the
    knot or level of each factor at each cell, and the names it drops.
    """
    factors = []
    for i in range(0, len(spec), 3):
        variable, points, reference = spec[i:i + 3]
        if isinstance(points[0], str):
            observed = frame[variable].astype(str).to_numpy()
            pieces = [(f"_{variable}{level}", level, (observed == level).astype(float)) for level in points]
        else:
            basis = reference_tent_basis(frame[variable], points)
            pieces = [(f"_{variable}{k}", knot, basis[:, k - 1]) for k, knot in enumerate(points, start=1)]
        positions = [] if reference is None else [reference] if isinstance(reference, int) else list(reference)
        factors.append((pieces, {points[position - 1] for position in positions}))
    values, points, dropped = {}, {}, []
    for combo in itertools.product(*[pieces for pieces, _ in factors]):
        name = "".join(piece[0] for piece in combo)
        values[name] = np.prod([piece[2] for piece in combo], axis=0)
        points[name] = tuple(piece[1] for piece in combo)
        if any(piece[1] in references for piece, (_, references) in zip(combo, factors)):
            dropped.append(name)
    return pd.DataFrame(values, index=frame.index), points, dropped


def hierarchy(*encodings, top=None) -> Design:
    """The design that holds the cross of ``encodings``: the intercept, each main term and every cross of them, each
    cross dropping the cells where any of its columns sits on its reference. ``top`` gives the highest cross encodings
    of its own, such as ones with further references."""
    terms = []
    for size in range(1, len(encodings) + 1):
        for combo in itertools.combinations(encodings, size):
            if size == len(encodings) and top is not None:
                combo = top
            terms.append(combo[0] if size == 1 else Cross(*combo, drop=[encoding.column for encoding in combo]))
    return Design(terms, intercept=True)


def reconcile(frame: pd.DataFrame, spec: tuple, design: Design, term: str):
    """One row per cell of ``spec``: whether each side keeps it, and how far apart the values are where both do.

    Cells are matched by their knots and levels, so the other implementation's names need no parsing. Also returns
    the columns of ``term`` that only ``design`` has.
    """
    reference, points, dropped = reference_cells(frame, spec)
    Z = design.fit_transform(frame)
    fitted = next(t for t in design.terms_ if t.name == term)
    encodings = fitted.encodings if isinstance(fitted, Cross) else [fitted]
    cells = dict(zip(itertools.product(*[encoding.points() for encoding in encodings]), fitted.columns()))
    kept = design.groups()[term]
    rows = []
    for name in reference.columns:
        column = cells[points[name]]
        both = name not in dropped and column in kept
        rows.append({"reference": name, "reference_keeps": name not in dropped, "design": column,
                     "design_keeps": column in kept,
                     "max_abs_diff": float(np.abs(reference[name] - Z[column]).max()) if both else np.nan})
    table = pd.DataFrame(rows)
    return table, [column for column in kept if column not in set(table["design"])]


def show(title: str, table: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_rows", 200,
                           "display.max_colwidth", 80):
        print(f"\n== {title}\n{table}")


def output_path(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def write_sheet(sheet: str, table: pd.DataFrame) -> Path:
    """Write ``table`` to ``sheet`` of the workbook, adding the sheet or replacing it and keeping the others.

    ``max_abs_diff`` is written as text in scientific notation, since a fixed number of decimals shows 1e-16 as 0.
    """
    path = output_path(WORKBOOK)
    table = table.assign(max_abs_diff=table["max_abs_diff"].map(lambda v: "" if pd.isna(v) else f"{v:.1e}"))
    schema = ColumnSchema([ColumnMeta(name=column, display_name=column, format=ColumnFormat(decimals=2))
                           for column in table.columns])
    ExcelExporter(str(path)).add_sheet(sheet, table, schema=schema).save()
    print(f"\nwritten: {path} [{sheet}]")
    return path


class TestDesignReconciliationExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Every example starts from the same fake loan-months.
        cls.loans = generate_fake_loan_months(n_rows=100_000, seed=0, incentive_range=INCENTIVE_RANGE)

    def test_tent_bases_match(self):
        """The tent columns of each continuous variable, including values exactly on the knots."""
        rows = []
        for column, knots in [("incentive", INCENTIVE_KNOTS), ("fico", FICO_KNOTS), ("upb", UPB_KNOTS)]:
            frame = pd.DataFrame({column: np.concatenate([self.loans[column].to_numpy(), knots])})
            design = Design([Tent(column, knots=knots, reference=None)],   # reference=None keeps every knot's column,
                            intercept=False)                               # so the tents carry the constant
            ours = design.fit_transform(frame).to_numpy()
            rows.append({"variable": column, "knots": ", ".join(f"{knot:g}" for knot in knots), "rows": len(frame),
                         "max_abs_diff": float(np.abs(ours - reference_tent_basis(frame[column], knots)).max())})
        table = pd.DataFrame(rows)
        show("tent bases, the other implementation against Design", table)
        path = write_sheet("tents", table)

        self.assertTrue((table["max_abs_diff"] <= TOLERANCE).all())
        self.assertIn("tents", pd.ExcelFile(path).sheet_names)

    def test_main_terms(self):
        """Purpose with cash-out as the reference, and incentive with knot 0 as the reference."""
        # The other implementation has no intercept: an offset carries the level, so no column moves it. Design builds
        # that as written and records the constraint: eta is 0 for cash-out loans, and at incentive 0.
        purpose = Design([OneHot("purpose", reference="C")],   # reference index 3: C
                         intercept=False)                      # no column of ones; True would add one
        incentive = Design([Tent("incentive", knots=INCENTIVE_KNOTS, reference=0.0)],   # reference index 2: knot 0
                           intercept=False)
        with self.assertWarns(DesignWarning):
            categorical, _ = reconcile(self.loans, ("purpose", PURPOSES, 3), purpose, term="purpose")
            continuous, _ = reconcile(self.loans, ("incentive", INCENTIVE_KNOTS, 2), incentive, term="incentive")
        print(f"Design's violation, without an intercept: {purpose.violations_[0]}")
        table = pd.concat([categorical, continuous], ignore_index=True)
        show("purpose and incentive, cell by cell", table)
        path = write_sheet("main terms", table)

        self.assertEqual([v.kind for v in purpose.violations_ + incentive.violations_], ["constraint", "constraint"])
        self.assertEqual(list(purpose.groups()), ["purpose"])                 # the other side's columns, no more
        self.assertTrue((table["reference_keeps"] == table["design_keeps"]).all())
        self.assertLessEqual(table["max_abs_diff"].max(), TOLERANCE)
        self.assertIn("main terms", pd.ExcelFile(path).sheet_names)

    def test_cat_x_cont(self):
        """No-cash-out refi times incentive, with incentive knot -1 as the reference."""
        spec = ("purpose", ["N"], None,           # one level, no reference: only N gets an incentive curve
                "incentive", INCENTIVE_KNOTS, 1)  # reference index 1: knot -1, unlike the main term's 0
        incentive = Tent("incentive", knots=INCENTIVE_KNOTS, reference=-1.0)
        only_n = OneHot("purpose", reference=["C", "P"])   # read from C, as the main term is; P joins it, so in the
                                                           # cross only N has cells of its own
        design = hierarchy(OneHot("purpose", reference="C"), incentive, top=(only_n, incentive))
        with self.assertWarns(DesignWarning):
            table, design_only = reconcile(self.loans, spec, design, term="purpose_x_incentive")
        print(f"Design's violation: {design.violations_[0]}")
        show("N x incentive, cell by cell", table)
        path = write_sheet("cat x cont", table)

        self.assertEqual([v.kind for v in design.violations_], ["constraint"])   # P's curve is the main term's
        self.assertTrue((table["reference_keeps"] == table["design_keeps"]).all())
        self.assertLessEqual(table["max_abs_diff"].max(), TOLERANCE)
        self.assertEqual(design_only, [])
        self.assertIn("cat x cont", pd.ExcelFile(path).sheet_names)

    def test_cont_x_cont(self):
        """Incentive times FICO, with knot 0 and 720 as the references."""
        incentive = Tent("incentive", knots=INCENTIVE_KNOTS, reference=0.0)
        fico = Tent("fico", knots=FICO_KNOTS, reference=720.0)
        # The other implementation's terms on these columns, as written: no intercept, the incentive main term, and the
        # cross with no FICO main term, which still drops its cells at FICO 720. Design builds them and records the
        # constant's constraint, that no drop of the cross fits a design without a constant, and what this drop
        # removes: the constant and FICO's own effect, so FICO has no effect at incentive 0.
        as_written = Design([incentive, Cross(incentive, fico, drop=["incentive", "fico"])], intercept=False)
        with self.assertWarns(DesignWarning):
            table, design_only = reconcile(self.loans, ("incentive", INCENTIVE_KNOTS, 2, "fico", FICO_KNOTS, 2),
                                           as_written, term="incentive_x_fico")
        for violation in as_written.violations_:
            print(f"\n{violation.kind} in {violation.term}:\n{violation}")
        show("incentive x FICO, cell by cell", table)
        path = write_sheet("cont x cont", table)

        self.assertEqual([(v.kind, v.term) for v in as_written.violations_],
                         [("constraint", "the constant"), ("hierarchy", "Cross('incentive_x_fico')"),
                          ("constraint", "Cross('incentive_x_fico')")])
        self.assertIn("the part that varies only with fico", as_written.violations_[-1].message)
        self.assertTrue((table["reference_keeps"] == table["design_keeps"]).all())
        self.assertLessEqual(table["max_abs_diff"].max(), TOLERANCE)
        self.assertEqual(design_only, [])
        self.assertIn("cont x cont", pd.ExcelFile(path).sheet_names)

    def test_cat_x_cont_x_cont(self):
        """No-cash-out refi times incentive times FICO, with knot 0 and 720 as the references."""
        spec = ("purpose", ["N"], None, "incentive", INCENTIVE_KNOTS, 2, "fico", FICO_KNOTS, 2)
        incentive = Tent("incentive", knots=INCENTIVE_KNOTS, reference=0.0)
        fico = Tent("fico", knots=FICO_KNOTS, reference=720.0)
        design = hierarchy(OneHot("purpose", reference="C"), incentive, fico,
                           top=(OneHot("purpose", reference=["C", "P"]), incentive, fico))   # only N in the triple
        with self.assertWarns(DesignWarning):
            table, design_only = reconcile(self.loans, spec, design, term="purpose_x_incentive_x_fico")
        show("N x incentive x FICO, cell by cell", table)
        path = write_sheet("cat x cont x cont", table)

        self.assertEqual([v.kind for v in design.violations_], ["constraint"])
        self.assertTrue((table["reference_keeps"] == table["design_keeps"]).all())
        self.assertLessEqual(table["max_abs_diff"].max(), TOLERANCE)
        self.assertEqual(design_only, [])
        self.assertIn("cat x cont x cont", pd.ExcelFile(path).sheet_names)

    def test_four_factor(self):
        """No-cash-out refi times incentive times FICO times UPB, with UPB 250K and 400K both as references.

        The four-way cross reads UPB from 250K and also drops its 400K cells, which holds its surface at 0 at 400K as
        well; UPB's main term and the lower crosses keep the one reference 250K, as in the other implementation.
        """
        spec = ("purpose", ["N"], None, "incentive", INCENTIVE_KNOTS, 2, "fico", FICO_KNOTS, 2,
                "upb", UPB_KNOTS, [2, 3])           # a list of references: knots 250K and 400K
        reference, _, dropped = reference_cells(self.loans, spec)
        print(f"The other implementation: {reference.shape[1]} cells, keeps {reference.shape[1] - len(dropped)}")

        incentive = Tent("incentive", knots=INCENTIVE_KNOTS, reference=0.0)
        fico = Tent("fico", knots=FICO_KNOTS, reference=720.0)
        design = hierarchy(OneHot("purpose", reference="C"), incentive, fico,
                           Tent("upb", knots=UPB_KNOTS, reference=250_000.0),
                           top=(OneHot("purpose", reference=["C", "P"]), incentive, fico,
                                Tent("upb", knots=UPB_KNOTS, reference=[250_000.0, 400_000.0])))
        with self.assertWarns(DesignWarning):
            table, design_only = reconcile(self.loans, spec, design, term="purpose_x_incentive_x_fico_x_upb")
        for violation in design.violations_:
            print(f"{violation.kind} in {violation.term}: {violation}")
        kept = table[table["reference_keeps"] | table["design_keeps"]]
        show("N x incentive x FICO x UPB, the cells either side keeps", kept)
        path = write_sheet("four factor", table)

        self.assertEqual([v.kind for v in design.violations_], ["constraint", "constraint"])   # P, and UPB 400K
        self.assertTrue((table["reference_keeps"] == table["design_keeps"]).all())
        self.assertEqual(len(kept), 3 * 3 * 2)                   # incentive 1, 3, 4 x FICO 1, 3, 4 x UPB 1, 4
        self.assertLessEqual(table["max_abs_diff"].max(), TOLERANCE)
        self.assertEqual(design_only, [])
        self.assertIn("four factor", pd.ExcelFile(path).sheet_names)
