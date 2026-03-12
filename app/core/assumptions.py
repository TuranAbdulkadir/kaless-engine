"""KALESS Engine — Assumption Checks.

Standard statistical assumption tests used across analyses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from app.schemas.results import AssumptionResult, AssumptionsBlock


def check_normality(
    series: pd.Series, var_name: str, alpha: float = 0.05
) -> AssumptionResult:
    """Shapiro-Wilk test for normality.

    Uses Shapiro-Wilk for n <= 5000, skips for larger samples.
    """
    clean = series.dropna()
    n = len(clean)

    if n < 3:
        return AssumptionResult(
            test_name="Shapiro-Wilk",
            description=f"Normality of '{var_name}'",
            passed=False,
            note="Sample too small for normality testing (n < 3).",
        )

    if n > 5000:
        return AssumptionResult(
            test_name="Shapiro-Wilk",
            description=f"Normality of '{var_name}'",
            passed=True,
            note=f"Skipped for large samples (n={n}). Central Limit Theorem applies.",
        )

    stat, p = stats.shapiro(clean)
    return AssumptionResult(
        test_name="Shapiro-Wilk",
        description=f"Normality of '{var_name}'",
        statistic=float(stat),
        p_value=float(p),
        passed=p > alpha,
        note=(
            f"Data appears normally distributed (W = {stat:.4f}, p = {p:.4f})."
            if p > alpha
            else f"Data departs from normality (W = {stat:.4f}, p = {p:.4f})."
        ),
    )


def check_homogeneity_of_variance(
    groups: dict[str, pd.Series], alpha: float = 0.05
) -> AssumptionResult:
    """Levene's test for equality of variances."""
    group_data = [g.dropna().values for g in groups.values() if len(g.dropna()) >= 2]

    if len(group_data) < 2:
        return AssumptionResult(
            test_name="Levene's Test",
            description="Homogeneity of variances",
            passed=False,
            note="Insufficient groups for Levene's test.",
        )

    stat, p = stats.levene(*group_data)
    return AssumptionResult(
        test_name="Levene's Test",
        description="Homogeneity of variances",
        statistic=float(stat),
        p_value=float(p),
        passed=p > alpha,
        note=(
            f"Equal variances assumed (F = {stat:.4f}, p = {p:.4f})."
            if p > alpha
            else f"Variances are significantly different (F = {stat:.4f}, p = {p:.4f})."
        ),
    )


def check_independence_expected_freq(
    expected: np.ndarray,
) -> AssumptionResult:
    """Check that expected cell frequencies are >= 5 for chi-square."""
    min_expected = float(expected.min())
    pct_below_5 = float((expected < 5).sum() / expected.size * 100)

    passed = min_expected >= 5
    note = f"Minimum expected frequency: {min_expected:.1f}."
    if not passed:
        note += f" {pct_below_5:.0f}% of cells have expected count < 5."

    return AssumptionResult(
        test_name="Expected Frequency Check",
        description="Chi-square minimum expected cell frequency",
        statistic=min_expected,
        passed=passed,
        note=note,
    )


def build_assumptions_block(
    checks: list[AssumptionResult],
) -> AssumptionsBlock:
    """Combine assumption checks into an AssumptionsBlock."""
    overall = all(c.passed for c in checks)
    warnings = [c.note for c in checks if not c.passed]
    return AssumptionsBlock(checks=checks, overall_passed=overall, warnings=warnings)
