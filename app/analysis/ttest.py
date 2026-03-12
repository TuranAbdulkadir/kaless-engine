"""KALESS Engine — T-Test Module.

Implements one-sample, independent samples, and paired t-tests.
Library: scipy.stats
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from app.core.preprocessing import (
    validate_variable_exists, validate_numeric, validate_exact_groups,
    validate_min_n, drop_missing_listwise, compute_descriptive, get_group_data,
)
from app.core.assumptions import check_normality, check_homogeneity_of_variance, build_assumptions_block
from app.core.effect_sizes import cohens_d_one_sample, cohens_d_independent, cohens_d_paired
from app.core.interpretation import determine_significance, interpret_ttest, format_p
from app.schemas.results import (
    NormalizedResult, PrimaryResult, GroupDescriptive,
    ConfidenceInterval,
)


def run_one_sample_ttest(
    df: pd.DataFrame,
    variable: str,
    test_value: float = 0.0,
    alpha: float = 0.05,
) -> NormalizedResult:
    """One-sample t-test against a known population mean."""
    start = time.time()
    warnings: list[str] = []

    validate_variable_exists(df, variable)
    validate_numeric(df[variable], variable)

    cleaned, n_dropped = drop_missing_listwise(df, [variable])
    if n_dropped > 0:
        warnings.append(f"{n_dropped} case(s) excluded due to missing values.")

    series = cleaned[variable]
    validate_min_n(len(series), 3, "one-sample t-test")

    # Assumptions
    normality = check_normality(series, variable, alpha)
    assumptions = build_assumptions_block([normality])
    if not normality.passed:
        warnings.append("Normality assumption not met. Results should be interpreted cautiously.")

    # Test
    t_stat, p_value = stats.ttest_1samp(series, test_value)
    df_val = len(series) - 1
    sig = determine_significance(float(p_value), alpha)

    # Descriptives
    desc = compute_descriptive(series, "Sample")
    mean_diff = desc["mean"] - test_value

    # CI of mean difference
    se = desc["se"]
    t_crit = stats.t.ppf(1 - alpha / 2, df_val)
    ci = ConfidenceInterval(
        lower=round(mean_diff - t_crit * se, 4),
        upper=round(mean_diff + t_crit * se, 4),
        level=1 - alpha,
    )

    # Effect size
    effect = cohens_d_one_sample(desc["mean"], test_value, desc["sd"])

    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type="one_sample_t_test",
        title=f"One-Sample T-Test — {variable}",
        variables={"dependent": variable, "test_value": str(test_value)},
        assumptions=assumptions,
        primary=PrimaryResult(
            statistic_name="t",
            statistic_value=round(float(t_stat), 4),
            df=float(df_val),
            p_value=round(float(p_value), 6),
            p_value_formatted=format_p(float(p_value)),
            significance=sig,
            alpha=alpha,
        ),
        descriptives=[GroupDescriptive(**desc)],
        effect_size=effect,
        confidence_interval=ci,
        interpretation=interpret_ttest(
            "one-sample t-test", float(t_stat), float(df_val), float(p_value), sig,
            effect.name, effect.value, effect.interpretation, variable,
        ),
        warnings=warnings,
        metadata={
            "n_total": len(df), "missing_excluded": n_dropped,
            "library": "scipy.stats", "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


def run_independent_ttest(
    df: pd.DataFrame,
    dependent: str,
    grouping: str,
    alpha: float = 0.05,
) -> NormalizedResult:
    """Independent samples t-test."""
    start = time.time()
    warnings: list[str] = []

    validate_variable_exists(df, dependent)
    validate_variable_exists(df, grouping)
    validate_numeric(df[dependent], dependent)

    cleaned, n_dropped = drop_missing_listwise(df, [dependent, grouping])
    if n_dropped > 0:
        warnings.append(f"{n_dropped} case(s) excluded due to missing values.")

    groups_list = validate_exact_groups(cleaned[grouping], grouping, 2)
    group_data = get_group_data(cleaned, dependent, grouping)

    for g_name, g_series in group_data.items():
        validate_min_n(len(g_series), 2, f"group '{g_name}'")

    # Assumptions
    normality_checks = [check_normality(g, f"{dependent} ({g_name})", alpha)
                        for g_name, g in group_data.items()]
    homogeneity = check_homogeneity_of_variance(group_data, alpha)
    assumptions = build_assumptions_block(normality_checks + [homogeneity])

    if not homogeneity.passed:
        warnings.append("Levene's test significant — using Welch's t-test (unequal variances).")

    # Descriptives per group
    descriptives = [GroupDescriptive(**compute_descriptive(g, g_name))
                    for g_name, g in group_data.items()]

    g1_series = group_data[groups_list[0]]
    g2_series = group_data[groups_list[1]]

    # Choose equal or Welch
    equal_var = homogeneity.passed
    t_stat, p_value = stats.ttest_ind(g1_series, g2_series, equal_var=equal_var)

    if equal_var:
        df_val = len(g1_series) + len(g2_series) - 2
    else:
        # Welch-Satterthwaite df
        s1, s2 = g1_series.var(ddof=1), g2_series.var(ddof=1)
        n1, n2 = len(g1_series), len(g2_series)
        num = (s1 / n1 + s2 / n2) ** 2
        den = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
        df_val = num / den if den > 0 else n1 + n2 - 2

    sig = determine_significance(float(p_value), alpha)

    # Effect size
    effect = cohens_d_independent(
        float(g1_series.mean()), float(g2_series.mean()),
        float(g1_series.std(ddof=1)), float(g2_series.std(ddof=1)),
        len(g1_series), len(g2_series),
    )

    # CI
    mean_diff = float(g1_series.mean() - g2_series.mean())
    se_diff = float(np.sqrt(g1_series.var(ddof=1) / len(g1_series) + g2_series.var(ddof=1) / len(g2_series)))
    t_crit = stats.t.ppf(1 - alpha / 2, df_val)
    ci = ConfidenceInterval(
        lower=round(mean_diff - t_crit * se_diff, 4),
        upper=round(mean_diff + t_crit * se_diff, 4),
        level=1 - alpha,
    )

    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type="independent_t_test",
        title=f"Independent Samples T-Test — {dependent} by {grouping}",
        variables={"dependent": dependent, "grouping": grouping, "groups": groups_list},
        assumptions=assumptions,
        primary=PrimaryResult(
            statistic_name="t",
            statistic_value=round(float(t_stat), 4),
            df=round(float(df_val), 2),
            p_value=round(float(p_value), 6),
            p_value_formatted=format_p(float(p_value)),
            significance=sig,
            alpha=alpha,
        ),
        descriptives=descriptives,
        effect_size=effect,
        confidence_interval=ci,
        interpretation=interpret_ttest(
            "independent samples t-test", float(t_stat), float(df_val), float(p_value), sig,
            effect.name, effect.value, effect.interpretation, dependent,
            f"between {groups_list[0]} and {groups_list[1]} ",
        ),
        warnings=warnings,
        metadata={
            "n_total": len(df), "missing_excluded": n_dropped,
            "library": "scipy.stats", "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


def run_paired_ttest(
    df: pd.DataFrame,
    variable1: str,
    variable2: str,
    alpha: float = 0.05,
) -> NormalizedResult:
    """Paired samples t-test."""
    start = time.time()
    warnings: list[str] = []

    validate_variable_exists(df, variable1)
    validate_variable_exists(df, variable2)
    validate_numeric(df[variable1], variable1)
    validate_numeric(df[variable2], variable2)

    cleaned, n_dropped = drop_missing_listwise(df, [variable1, variable2])
    if n_dropped > 0:
        warnings.append(f"{n_dropped} case(s) excluded due to missing values.")

    s1 = cleaned[variable1]
    s2 = cleaned[variable2]
    validate_min_n(len(s1), 3, "paired t-test")

    diff = s1 - s2

    # Assumptions
    normality = check_normality(diff, "differences", alpha)
    assumptions = build_assumptions_block([normality])
    if not normality.passed:
        warnings.append("Normality of differences not met. Results should be interpreted cautiously.")

    # Descriptives
    desc1 = GroupDescriptive(**compute_descriptive(s1, variable1))
    desc2 = GroupDescriptive(**compute_descriptive(s2, variable2))
    diff_desc = compute_descriptive(diff, "Difference")

    # Test
    t_stat, p_value = stats.ttest_rel(s1, s2)
    df_val = len(s1) - 1
    sig = determine_significance(float(p_value), alpha)

    # Effect size
    effect = cohens_d_paired(diff_desc["mean"], diff_desc["sd"])

    # CI
    se_diff = diff_desc["se"]
    t_crit = stats.t.ppf(1 - alpha / 2, df_val)
    ci = ConfidenceInterval(
        lower=round(diff_desc["mean"] - t_crit * se_diff, 4),
        upper=round(diff_desc["mean"] + t_crit * se_diff, 4),
        level=1 - alpha,
    )

    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type="paired_t_test",
        title=f"Paired Samples T-Test — {variable1} vs {variable2}",
        variables={"variable1": variable1, "variable2": variable2},
        assumptions=assumptions,
        primary=PrimaryResult(
            statistic_name="t",
            statistic_value=round(float(t_stat), 4),
            df=float(df_val),
            p_value=round(float(p_value), 6),
            p_value_formatted=format_p(float(p_value)),
            significance=sig,
            alpha=alpha,
        ),
        descriptives=[desc1, desc2, GroupDescriptive(**diff_desc)],
        effect_size=effect,
        confidence_interval=ci,
        interpretation=interpret_ttest(
            "paired samples t-test", float(t_stat), float(df_val), float(p_value), sig,
            effect.name, effect.value, effect.interpretation, f"{variable1} and {variable2}",
        ),
        warnings=warnings,
        metadata={
            "n_total": len(df), "missing_excluded": n_dropped,
            "library": "scipy.stats", "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
