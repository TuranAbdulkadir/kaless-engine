"""KALESS Engine — Standalone Test Runner.

Uses Python's built-in unittest (no pytest required).
Tests all 7 analysis families:  descriptives, t-tests, ANOVA, chi-square,
correlation, regression + registry + NormalizedResult shape.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Ensure the engine root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.results import NormalizedResult, SignificanceLevel
from app.analysis.descriptives import run_descriptives, run_frequencies
from app.analysis.ttest import run_one_sample_ttest, run_independent_ttest, run_paired_ttest
from app.analysis.anova import run_one_way_anova
from app.analysis.chi_square import run_chi_square_independence
from app.analysis.correlation import run_pearson_correlation, run_spearman_correlation
from app.analysis.regression import run_linear_regression
from app.analysis.registry import dispatch_analysis, get_available_analyses
from app.utils.errors import ValidationError


def make_exam_data():
    np.random.seed(42)
    n = 50
    gender = np.random.choice(["Male", "Female"], n)
    scores = np.where(gender == "Male", np.random.normal(78, 10, n), np.random.normal(72, 10, n))
    grade = pd.cut(scores, bins=[0, 65, 80, 100], labels=["C", "B", "A"])
    return pd.DataFrame({
        "math_score": scores.round(1), "reading_score": (scores + np.random.normal(0, 5, n)).round(1),
        "gender": gender, "grade": grade.astype(str),
    })


def make_paired_data():
    np.random.seed(123)
    n = 30
    pre = np.random.normal(50, 8, n).round(1)
    post = (pre + np.random.normal(5, 3, n)).round(1)
    return pd.DataFrame({"pre_test": pre, "post_test": post})


def make_anova_data():
    np.random.seed(99)
    groups = np.repeat(["Control", "Treatment_A", "Treatment_B"], 20)
    scores = np.concatenate([np.random.normal(50, 8, 20), np.random.normal(55, 8, 20), np.random.normal(62, 8, 20)]).round(1)
    return pd.DataFrame({"group": groups, "score": scores})


def make_categorical_data():
    np.random.seed(77)
    n = 100
    smoker = np.random.choice(["Yes", "No"], n, p=[0.4, 0.6])
    lung_disease = np.where(smoker == "Yes",
        np.random.choice(["Yes", "No"], n, p=[0.6, 0.4]),
        np.random.choice(["Yes", "No"], n, p=[0.15, 0.85]))
    return pd.DataFrame({"smoker": smoker, "lung_disease": lung_disease})


def make_regression_data():
    np.random.seed(55)
    n = 50
    x1 = np.random.uniform(0, 10, n)
    x2 = np.random.uniform(0, 5, n)
    y = 2 * x1 + 3 * x2 + np.random.normal(0, 2, n)
    return pd.DataFrame({"y": y.round(2), "x1": x1.round(2), "x2": x2.round(2)})


class TestDescriptives(unittest.TestCase):
    def setUp(self):
        self.df = make_exam_data()

    def test_basic(self):
        r = run_descriptives(self.df, ["math_score"])
        self.assertIsInstance(r, NormalizedResult)
        self.assertEqual(r.analysis_type, "descriptives")
        self.assertEqual(len(r.descriptives), 1)
        self.assertEqual(r.descriptives[0].n, 50)
        self.assertIsNotNone(r.descriptives[0].mean)

    def test_multi_var(self):
        r = run_descriptives(self.df, ["math_score", "reading_score"])
        self.assertEqual(len(r.descriptives), 2)

    def test_chart(self):
        r = run_descriptives(self.df, ["math_score"])
        self.assertTrue(len(r.charts) >= 1)

    def test_missing_var(self):
        with self.assertRaises(ValidationError):
            run_descriptives(self.df, ["nonexistent"])

    def test_non_numeric(self):
        with self.assertRaises(ValidationError):
            run_descriptives(self.df, ["gender"])


class TestFrequencies(unittest.TestCase):
    def test_basic(self):
        r = run_frequencies(make_exam_data(), "grade")
        self.assertEqual(r.analysis_type, "frequencies")
        self.assertTrue(len(r.frequency_table) > 0)
        self.assertEqual(sum(x["frequency"] for x in r.frequency_table), 50)


class TestOneSampleTTest(unittest.TestCase):
    def setUp(self):
        self.df = make_exam_data()

    def test_basic(self):
        r = run_one_sample_ttest(self.df, "math_score", test_value=70)
        self.assertEqual(r.analysis_type, "one_sample_t_test")
        self.assertIsNotNone(r.primary)
        self.assertEqual(r.primary.statistic_name, "t")
        self.assertIsNotNone(r.effect_size)
        self.assertIsNotNone(r.confidence_interval)
        self.assertIsNotNone(r.interpretation)

    def test_normality_assumption(self):
        r = run_one_sample_ttest(self.df, "math_score", test_value=70)
        self.assertIsNotNone(r.assumptions)
        self.assertTrue(len(r.assumptions.checks) > 0)


class TestIndependentTTest(unittest.TestCase):
    def setUp(self):
        self.df = make_exam_data()

    def test_basic(self):
        r = run_independent_ttest(self.df, "math_score", "gender")
        self.assertEqual(r.analysis_type, "independent_t_test")
        self.assertEqual(len(r.descriptives), 2)

    def test_wrong_groups(self):
        with self.assertRaises(ValidationError):
            run_independent_ttest(make_anova_data(), "score", "group")  # 3 groups


class TestPairedTTest(unittest.TestCase):
    def test_basic(self):
        r = run_paired_ttest(make_paired_data(), "pre_test", "post_test")
        self.assertEqual(r.analysis_type, "paired_t_test")
        self.assertEqual(len(r.descriptives), 3)

    def test_detects_improvement(self):
        r = run_paired_ttest(make_paired_data(), "pre_test", "post_test")
        self.assertLess(r.primary.p_value, 0.05)
        self.assertEqual(r.primary.significance, SignificanceLevel.SIGNIFICANT)


class TestANOVA(unittest.TestCase):
    def test_basic(self):
        r = run_one_way_anova(make_anova_data(), "score", "group")
        self.assertEqual(r.analysis_type, "one_way_anova")
        self.assertEqual(r.primary.statistic_name, "F")
        self.assertIsNotNone(r.primary.df2)

    def test_significant(self):
        r = run_one_way_anova(make_anova_data(), "score", "group")
        self.assertLess(r.primary.p_value, 0.05)

    def test_post_hoc(self):
        r = run_one_way_anova(make_anova_data(), "score", "group")
        self.assertTrue(len(r.post_hoc) > 0)

    def test_eta_squared(self):
        r = run_one_way_anova(make_anova_data(), "score", "group")
        self.assertIsNotNone(r.effect_size)
        self.assertIn("eta", r.effect_size.name.lower())


class TestChiSquare(unittest.TestCase):
    def test_basic(self):
        r = run_chi_square_independence(make_categorical_data(), "smoker", "lung_disease")
        self.assertEqual(r.analysis_type, "chi_square_independence")
        self.assertEqual(r.primary.statistic_name, "χ²")

    def test_crosstab(self):
        r = run_chi_square_independence(make_categorical_data(), "smoker", "lung_disease")
        self.assertIn("observed", r.crosstab)
        self.assertIn("expected", r.crosstab)

    def test_cramers_v(self):
        r = run_chi_square_independence(make_categorical_data(), "smoker", "lung_disease")
        self.assertEqual(r.effect_size.name, "Cramér's V")


class TestCorrelation(unittest.TestCase):
    def test_pearson(self):
        r = run_pearson_correlation(make_exam_data(), "math_score", "reading_score")
        self.assertEqual(r.analysis_type, "pearson_correlation")
        self.assertEqual(r.primary.statistic_name, "r")
        self.assertTrue(-1 <= r.primary.statistic_value <= 1)
        self.assertIsNotNone(r.confidence_interval)

    def test_spearman(self):
        r = run_spearman_correlation(make_exam_data(), "math_score", "reading_score")
        self.assertEqual(r.primary.statistic_name, "ρ")

    def test_positive_correlation(self):
        r = run_pearson_correlation(make_exam_data(), "math_score", "reading_score")
        self.assertGreater(r.primary.statistic_value, 0)


class TestRegression(unittest.TestCase):
    def test_simple(self):
        r = run_linear_regression(make_regression_data(), "y", ["x1"])
        self.assertEqual(r.analysis_type, "linear_regression")
        self.assertEqual(r.primary.statistic_name, "F")

    def test_multiple(self):
        r = run_linear_regression(make_regression_data(), "y", ["x1", "x2"])
        self.assertEqual(len(r.coefficients), 3)  # Constant + x1 + x2

    def test_coefficient_values(self):
        r = run_linear_regression(make_regression_data(), "y", ["x1", "x2"])
        x1c = next(c for c in r.coefficients if c.name == "x1")
        x2c = next(c for c in r.coefficients if c.name == "x2")
        self.assertAlmostEqual(x1c.b, 2.0, delta=1.0)
        self.assertAlmostEqual(x2c.b, 3.0, delta=1.5)

    def test_model_summary(self):
        r = run_linear_regression(make_regression_data(), "y", ["x1", "x2"])
        self.assertIn("r_squared", r.model_summary)
        self.assertGreater(r.model_summary["r_squared"], 0.5)

    def test_vif(self):
        r = run_linear_regression(make_regression_data(), "y", ["x1", "x2"])
        vif_coefs = [c for c in r.coefficients if c.vif is not None]
        self.assertEqual(len(vif_coefs), 2)

    def test_no_predictors(self):
        with self.assertRaises(ValidationError):
            run_linear_regression(make_regression_data(), "y", [])


class TestRegistry(unittest.TestCase):
    def test_available(self):
        analyses = get_available_analyses()
        self.assertGreaterEqual(len(analyses), 11)
        types = [a["analysis_type"] for a in analyses]
        self.assertIn("descriptives", types)
        self.assertIn("linear_regression", types)

    def test_dispatch(self):
        r = dispatch_analysis("descriptives", make_exam_data(), {"variables": ["math_score"]})
        self.assertIsInstance(r, NormalizedResult)

    def test_unknown(self):
        with self.assertRaises(ValidationError):
            dispatch_analysis("unknown", make_exam_data(), {})


class TestNormalizedResultShape(unittest.TestCase):
    """All analyses must return valid NormalizedResult with required fields."""

    def test_all_analyses_return_valid_result(self):
        results = [
            run_descriptives(make_exam_data(), ["math_score"]),
            run_frequencies(make_exam_data(), "grade"),
            run_one_sample_ttest(make_exam_data(), "math_score", 70),
            run_independent_ttest(make_exam_data(), "math_score", "gender"),
            run_paired_ttest(make_paired_data(), "pre_test", "post_test"),
            run_one_way_anova(make_anova_data(), "score", "group"),
            run_chi_square_independence(make_categorical_data(), "smoker", "lung_disease"),
            run_pearson_correlation(make_exam_data(), "math_score", "reading_score"),
            run_spearman_correlation(make_exam_data(), "math_score", "reading_score"),
            run_linear_regression(make_regression_data(), "y", ["x1", "x2"]),
        ]
        for r in results:
            self.assertIsInstance(r, NormalizedResult, f"Failed for {r.analysis_type}")
            self.assertNotEqual(r.analysis_type, "")
            self.assertNotEqual(r.title, "")
            self.assertNotEqual(r.variables, {})
            self.assertIsInstance(r.warnings, list)
            self.assertIn("n_total", r.metadata)
            self.assertIn("library", r.metadata)
            self.assertIn("duration_ms", r.metadata)


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
