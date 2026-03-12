"""KALESS Engine — Correlation Module.

Implements Pearson and Spearman correlations.
Library: scipy.stats
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from app.core.preprocessing import (
    validate_variable_exists, validate_numeric, validate_min_n,
    drop_missing_listwise, compute_descriptive,
)
from app.core.assumptions import check_normality, build_assumptions_block
from app.core.effect_sizes import r_effect_size
from app.core.interpretation import determine_significance, interpret_correlation, format_p
from app.schemas.results import (
    NormalizedResult, PrimaryResult, GroupDescriptive,
    ConfidenceInterval, ChartData,
)


def run_pearson_correlation(
    df: pd.DataFrame,
    variable1: str,
    variable2: str,
    alpha: float = 0.05,
) -> NormalizedResult:
    """Pearson product-moment correlation."""
    return _run_correlation(df, variable1, variable2, method="pearson", alpha=alpha)


def run_spearman_correlation(
    df: pd.DataFrame,
    variable1: str,
    variable2: str,
    alpha: float = 0.05,
) -> NormalizedResult:
    """Spearman rank correlation."""
    return _run_correlation(df, variable1, variable2, method="spearman", alpha=alpha)


def run_correlation_matrix(
    df: pd.DataFrame,
    variables: list[str],
    method: str = "pearson",
    alpha: float = 0.05,
) -> NormalizedResult:
    """Correlation matrix for multiple variables."""
    start = time.time()
    warnings: list[str] = []

    for v in variables:
        validate_variable_exists(df, v)
        validate_numeric(df[v], v)

    cleaned, n_dropped = drop_missing_listwise(df, variables)
    if n_dropped > 0:
        warnings.append(f"{n_dropped} case(s) excluded due to missing values.")

    n = len(cleaned)
    validate_min_n(n, 3, "correlation matrix")

    # Compute matrix
    matrix_data: list[dict] = []
    for v1 in variables:
        row: dict = {"variable": v1}
        for v2 in variables:
            if method == "pearson":
                r, p = stats.pearsonr(cleaned[v1], cleaned[v2])
            else:
                r, p = stats.spearmanr(cleaned[v1], cleaned[v2])
            row[v2] = round(float(r), 4)
            row[f"{v2}_p"] = round(float(p), 6)
        matrix_data.append(row)

    # Descriptives
    descriptives = [GroupDescriptive(**compute_descriptive(cleaned[v], v)) for v in variables]

    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type=f"{method}_correlation_matrix",
        title=f"{method.capitalize()} Correlation Matrix",
        variables={"analyzed": variables},
        correlation_matrix=matrix_data,
        descriptives=descriptives,
        warnings=warnings,
        metadata={
            "n_total": len(df), "missing_excluded": n_dropped,
            "library": "scipy.stats", "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


def _run_correlation(
    df: pd.DataFrame,
    variable1: str,
    variable2: str,
    method: str = "pearson",
    alpha: float = 0.05,
) -> NormalizedResult:
    """Internal: runs a bivariate correlation."""
    start = time.time()
    warnings: list[str] = []

    validate_variable_exists(df, variable1)
    validate_variable_exists(df, variable2)
    validate_numeric(df[variable1], variable1)
    validate_numeric(df[variable2], variable2)

    cleaned, n_dropped = drop_missing_listwise(df, [variable1, variable2])
    if n_dropped > 0:
        warnings.append(f"{n_dropped} case(s) excluded due to missing values.")

    n = len(cleaned)
    validate_min_n(n, 3, f"{method} correlation")

    s1 = cleaned[variable1]
    s2 = cleaned[variable2]

    # Assumptions (normality for Pearson)
    assumption_checks = []
    if method == "pearson":
        assumption_checks.append(check_normality(s1, variable1, alpha))
        assumption_checks.append(check_normality(s2, variable2, alpha))
        if any(not c.passed for c in assumption_checks):
            warnings.append("Normality assumption not met. Consider using Spearman correlation.")

    assumptions = build_assumptions_block(assumption_checks) if assumption_checks else None

    # Compute correlation
    if method == "pearson":
        r, p_value = stats.pearsonr(s1, s2)
        stat_name = "r"
    else:
        r, p_value = stats.spearmanr(s1, s2)
        stat_name = "ρ"

    sig = determine_significance(float(p_value), alpha)

    # CI for r using Fisher z-transformation
    ci = None
    if n > 3:
        z = np.arctanh(r)
        se_z = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci = ConfidenceInterval(
            lower=round(float(np.tanh(z - z_crit * se_z)), 4),
            upper=round(float(np.tanh(z + z_crit * se_z)), 4),
            level=1 - alpha,
        )

    # Effect size
    effect = r_effect_size(float(r))

    # Descriptives
    desc1 = GroupDescriptive(**compute_descriptive(s1, variable1))
    desc2 = GroupDescriptive(**compute_descriptive(s2, variable2))

    # Scatter chart
    scatter_data = [
        {variable1: float(s1.iloc[i]), variable2: float(s2.iloc[i])}
        for i in range(min(n, 500))  # Cap at 500 points for performance
    ]

    duration = int((time.time() - start) * 1000)

    method_label = "Pearson" if method == "pearson" else "Spearman"

    return NormalizedResult(
        analysis_type=f"{method}_correlation",
        title=f"{method_label} Correlation — {variable1} × {variable2}",
        variables={"variable1": variable1, "variable2": variable2},
        assumptions=assumptions,
        primary=PrimaryResult(
            statistic_name=stat_name,
            statistic_value=round(float(r), 4),
            df=float(n - 2),
            p_value=round(float(p_value), 6),
            p_value_formatted=format_p(float(p_value)),
            significance=sig,
            alpha=alpha,
        ),
        descriptives=[desc1, desc2],
        effect_size=effect,
        confidence_interval=ci,
        charts=[ChartData(
            chart_type="scatter",
            data=scatter_data,
            config={"title": f"{variable1} vs {variable2}", "xLabel": variable1, "yLabel": variable2},
        )],
        interpretation=interpret_correlation(
            method_label, float(r), float(p_value), sig,
            effect.interpretation, variable1, variable2,
        ),
        warnings=warnings,
        metadata={
            "n_total": len(df), "missing_excluded": n_dropped,
            "library": "scipy.stats", "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
