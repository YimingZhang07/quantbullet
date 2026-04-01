import os
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from quantbullet.model import WrapperGAM
from quantbullet.model.gam_replay import GAMReplayModel
from quantbullet.model.feature import FeatureSpec, FeatureRole, Feature
from quantbullet.core.enums import DataType
from quantbullet.model.gam import (
    SplineTermData,
    SplineByGroupTermData,
    TensorTermData,
    FactorTermData,
    center_partial_dependence,
)

DEV_MODE = True
CACHE_DIR = "./tests/_cache_dir"


# ============================================================================
# Shared fixtures
# ============================================================================

def _make_test_data(n_samples=200, seed=42):
    np.random.seed(seed)
    age = np.random.uniform(20, 80, n_samples)
    income = np.random.uniform(20000, 120000, n_samples)
    education = np.random.uniform(8, 20, n_samples)
    level = np.random.choice(["highschool", "bachelor", "master", "phd"], n_samples)

    happiness = (
        0.5 * np.sin((age - 40) / 10)
        + 0.3 * np.log(income / 30000)
        + 0.2 * education
        + 0.1 * (level == "phd").astype(float)
        + np.random.normal(0, 0.5, n_samples)
    )

    return pd.DataFrame({
        "age": age,
        "income": income,
        "education": education,
        "level": level,
        "happiness": happiness,
    })


def _make_feature_spec():
    """Covers all term types: spline-by-group, constrained spline, tensor, factor."""
    return FeatureSpec(features=[
        Feature(name="age", dtype=DataType.FLOAT, role=FeatureRole.MODEL_INPUT,
                specs={"spline_order": 3, "n_splines": 6, "lam": 0.1, "by": "level"}),
        Feature(name="income", dtype=DataType.FLOAT, role=FeatureRole.MODEL_INPUT,
                specs={"spline_order": 3, "n_splines": 6, "lam": 0.1, "constraints": "monotonic_inc"}),
        Feature(name="education", dtype=DataType.FLOAT, role=FeatureRole.MODEL_INPUT,
                specs={"spline_order": 3, "n_splines": 6, "lam": 0.1, "by": "income"}),
        Feature(name="level", dtype=DataType.CATEGORY, role=FeatureRole.MODEL_INPUT),
        Feature(name="happiness", dtype=DataType.FLOAT, role=FeatureRole.TARGET),
    ])


def _cache_path(filename):
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return os.path.join(CACHE_DIR, filename)


# ============================================================================
# 1. Core WrapperGAM tests
# ============================================================================

class TestWrapperGAM(unittest.TestCase):
    """Fit, predict, partial-dependence structure, plotting."""

    @classmethod
    def setUpClass(cls):
        cls.data = _make_test_data()
        cls.features = _make_feature_spec()
        cls.wgam = WrapperGAM(cls.features)
        cls.wgam.fit(cls.data, cls.data["happiness"])

    def test_fit_and_predict(self):
        self.assertIsNotNone(self.wgam.gam_)
        self.assertIn("level", self.wgam.category_levels_)

        preds = self.wgam.predict(self.data)
        self.assertEqual(len(preds), len(self.data))
        self.assertFalse(np.any(np.isnan(preds)))

    def test_partial_dependence_structure(self):
        pdata = self.wgam.get_partial_dependence_data()

        self.assertIn("income", pdata)
        self.assertIsInstance(pdata["income"], SplineTermData)
        self.assertIsNotNone(pdata["income"].conf_lower)

        self.assertIn(("age", "level"), pdata)
        self.assertIsInstance(pdata[("age", "level")], SplineByGroupTermData)
        self.assertIn("bachelor", pdata[("age", "level")].group_curves)

        self.assertIn(("education", "income"), pdata)
        self.assertIsInstance(pdata[("education", "income")], TensorTermData)

        self.assertIn("level", pdata)
        self.assertIsInstance(pdata["level"], FactorTermData)
        self.assertEqual(
            set(pdata["level"].categories),
            {"highschool", "bachelor", "master", "phd"},
        )

    def test_plot_smoke(self):
        fig, _ = self.wgam.plot_partial_dependence(
            scale_y_axis=False, te_plot_style="contourf",
        )
        plt.close(fig)


# ============================================================================
# 2. Replay model tests (raw / uncentered)
# ============================================================================

class TestReplayModel(unittest.TestCase):
    """GAMReplayModel accuracy against WrapperGAM (no centering)."""

    @classmethod
    def setUpClass(cls):
        cls.data = _make_test_data()
        cls.wgam = WrapperGAM(_make_feature_spec())
        cls.wgam.fit(cls.data, cls.data["happiness"])
        cls.original_preds = cls.wgam.predict(cls.data)

    def test_replay_prediction(self):
        pdata = self.wgam.get_partial_dependence_data()
        replay = GAMReplayModel(pdata, intercept=self.wgam.intercept_)
        np.testing.assert_allclose(
            self.original_preds, replay.predict(self.data),
            rtol=0.01, atol=0.05,
        )

    def test_replay_extrapolation_safety(self):
        pdata = self.wgam.get_partial_dependence_data()
        replay = GAMReplayModel(pdata, intercept=self.wgam.intercept_)

        oob = self.data.iloc[:5].copy()
        oob["income"] = 1_000_000.0
        oob["age"] = 150.0

        preds = replay.predict(oob)
        self.assertEqual(len(preds), 5)
        self.assertFalse(np.any(np.isnan(preds)))
        self.assertFalse(np.any(np.isinf(preds)))

    def test_json_roundtrip(self):
        path = _cache_path("test_gam_pdep.json") if DEV_MODE else \
            os.path.join(tempfile.mkdtemp(), "pdep.json")

        self.wgam.export_partial_dependence_json(path)
        replay = GAMReplayModel.from_partial_dependence_json(path)
        np.testing.assert_allclose(
            self.original_preds, replay.predict(self.data),
            rtol=0.01, atol=0.05,
        )

    def test_decompose_matches(self):
        sample = self.data.iloc[:10]
        orig = self.wgam.decompose(sample)

        pdata = self.wgam.get_partial_dependence_data()
        replay = GAMReplayModel(pdata, intercept=self.wgam.intercept_)
        rep = replay.decompose(sample)

        self.assertAlmostEqual(orig["intercept"], rep["intercept"])
        np.testing.assert_allclose(orig["pred"], rep["pred"], rtol=0.01, atol=0.05)

        self.assertEqual(set(orig["term_contrib"].columns),
                         set(rep["term_contrib"].columns))
        for col in orig["term_contrib"].columns:
            np.testing.assert_allclose(
                orig["term_contrib"][col], rep["term_contrib"][col],
                rtol=0.01, atol=0.05, err_msg=f"Mismatch in column: {col}",
            )


# ============================================================================
# 3. Centering tests
# ============================================================================

class TestCentering(unittest.TestCase):
    """Partial-dependence centering: invariance, replay accuracy, property."""

    @classmethod
    def setUpClass(cls):
        cls.data = _make_test_data()
        cls.wgam = WrapperGAM(_make_feature_spec())
        cls.wgam.fit(cls.data, cls.data["happiness"])
        cls.original_preds = cls.wgam.predict(cls.data)

    # -- prediction invariance ------------------------------------------------

    def test_centered_replay_prediction(self):
        """Centered data + centered_intercept_ -> predictions match original."""
        pdata = self.wgam.get_partial_dependence_data(center=True)
        replay = GAMReplayModel(pdata, intercept=self.wgam.centered_intercept_)
        np.testing.assert_allclose(
            self.original_preds, replay.predict(self.data),
            rtol=0.01, atol=0.05,
        )

    def test_centered_json_roundtrip(self):
        """Centered JSON export -> load -> replay reproduces predictions."""
        path = _cache_path("test_gam_pdep_centered.json") if DEV_MODE else \
            os.path.join(tempfile.mkdtemp(), "pdep_centered.json")

        self.wgam.export_partial_dependence_json(path, center=True)
        replay = GAMReplayModel.from_partial_dependence_json(path)
        np.testing.assert_allclose(
            self.original_preds, replay.predict(self.data),
            rtol=0.01, atol=0.05,
        )

    # -- curve properties -----------------------------------------------------

    def test_spline_mean_is_zero(self):
        """After centering, every simple spline's y values average to ~0."""
        centered = self.wgam.get_partial_dependence_data(center=True)
        for key, td in centered.items():
            if isinstance(td, SplineTermData):
                self.assertAlmostEqual(
                    float(np.mean(td.y)), 0.0, places=10,
                    msg=f"Spline '{td.feature}' mean not zero after centering",
                )

    def test_by_group_means_are_zero(self):
        """After centering, each by-group curve's y values average to ~0."""
        centered = self.wgam.get_partial_dependence_data(center=True)
        for key, td in centered.items():
            if isinstance(td, SplineByGroupTermData):
                for label, curves in td.group_curves.items():
                    self.assertAlmostEqual(
                        float(np.mean(curves["y"])), 0.0, places=10,
                        msg=f"By-group curve '{td.feature}' group='{label}' "
                            f"mean not zero after centering",
                    )

    def test_factor_mean_is_zero(self):
        """After centering, every factor term (including those that absorbed
        by-group offsets) averages to ~0."""
        centered = self.wgam.get_partial_dependence_data(center=True)
        for key, td in centered.items():
            if isinstance(td, FactorTermData):
                self.assertAlmostEqual(
                    float(np.mean(td.values)), 0.0, places=10,
                    msg=f"Factor '{td.feature}' mean not zero after centering",
                )

    # -- intercept property ---------------------------------------------------

    def test_centered_intercept_is_finite(self):
        self.assertTrue(np.isfinite(self.wgam.centered_intercept_))

    def test_centered_intercept_is_cached(self):
        val1 = self.wgam.centered_intercept_
        val2 = self.wgam.centered_intercept_
        self.assertEqual(val1, val2)

    # -- visual comparison (DEV_MODE only) ------------------------------------

    def test_centering_comparison_pdf(self):
        """Generate PDF with raw vs centered partial dependence side-by-side."""
        if not DEV_MODE:
            self.skipTest("PDF comparison only in DEV_MODE")

        pdf_path = _cache_path("centering_comparison.pdf")

        with PdfPages(pdf_path) as pdf:
            fig, _ = self.wgam.plot_partial_dependence(
                center=False, suptitle="Raw Partial Dependence",
                scale_y_axis=False,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

            fig, _ = self.wgam.plot_partial_dependence(
                center=True, suptitle="Centered Partial Dependence",
                scale_y_axis=False,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
