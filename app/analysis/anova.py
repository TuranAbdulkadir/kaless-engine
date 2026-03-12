"""KALESS Engine — ANOVA Module.

Implements one-way ANOVA with Bonferroni-corrected pairwise post-hoc.
Library: scipy.stats
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from app.core.preprocessing import (
    validate_variable_exists, validate_numeric, validate_min_groups,
    validate_min_n, drop_missing_listwise, compute_descriptive, get_group_data,
)
from app.core.assumptions import check_normality, check_homogeneity_of_variance, build_assumptions_block
from app.core.effect_sizes import eta_squared
from app.core.interpretation import determine_significance, interpret_anova, format_p
from app.schemas.results import (
    NormalizedResult, PrimaryResult, GroupDescriptive,
    PostHocPair, ChartData,
)


def run_one_way_anova(
    df: pd.DataFrame,
    dependent: str,
    grouping: str,
    alpha: float = 0.05,
    post_hoc: bool = True,
) -> NormalizedResult:
    """One-way between-subjects ANOVA."""
    start = time.time()
    warnings: list[str] = []

    validate_variable_exists(df, dependent)
    validate_variable_exists(df, grouping)
    validate_numeric(df[dependent], dependent)

    cleaned, n_dropped = drop_missing_listwise(df, [dependent, grouping])
    if n_dropped > 0:
        warnings.append(f"{n_dropped} case(s) excluded due to missing values.")

    groups_list = validate_min_groups(cleaned[grouping], grouping, 2)
    group_data = get_group_data(cleaned, dependent, grouping)

    for g_name, g_series in group_data.items():
        validate_min_n(len(g_series), 2, f"group '{g_name}'")

    # Assumptions
    normality_checks = [check_normality(g, f"{dependent} ({g_name})", alpha)
                        for g_name, g in group_data.items()]
    homogeneity = check_homogeneity_of_variance(group_data, alpha)
    assumptions = build_assumptions_block(normality_checks + [homogeneity])

    if not homogeneity.passed:
        warnings.append("Levene's test significant — consider Welch's ANOVA for unequal variances.")

    # Descriptives per group
    descriptives = [GroupDescriptive(**compute_descriptive(g, g_name))
                    for g_name, g in group_data.items()]
    # Overall descriptive
    descriptives.append(GroupDescriptive(**compute_descriptive(cleaned[dependent], "Total")))

    # One-way ANOVA
    group_arrays = [g.values for g in group_data.values()]
    f_stat, p_value = stats.f_oneway(*group_arrays)

    # Compute SS for effect size
    grand_mean = float(cleaned[dependent].mean())
    ss_between = sum(len(g) * (float(g.mean()) - grand_mean) ** 2 for g in group_data.values())
    ss_total = float(((cleaned[dependent] - grand_mean) ** 2).sum())

    k = len(group_data)
    n_total = len(cleaned)
    df_between = k - 1
    df_within = n_total - k
    sig = determine_significance(float(p_value), alpha)

    # Effect size
    effect = eta_squared(ss_between, ss_total)

    # Post-hoc (Bonferroni-corrected pairwise t-tests)
    post_hoc_results: list[PostHocPair] = []
    if post_hoc and float(p_value) <= alpha:
        group_names = list(group_data.keys())
        n_comparisons = len(group_names) * (len(group_names) - 1) // 2
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                g1 = group_data[group_names[i]]
                g2 = group_data[group_names[j]]
                t, p_pair = stats.ttest_ind(g1, g2, equal_var=homogeneity.passed)
                p_adj = min(float(p_pair) * n_comparisons, 1.0)  # Bonferroni
                mean_diff = float(g1.mean() - g2.mean())
                se = float(np.sqrt(g1.var(ddof=1) / len(g1) + g2.var(ddof=1) / len(g2)))
                t_crit = stats.t.ppf(1 - alpha / (2 * n_comparisons), len(g1) + len(g2) - 2)
                post_hoc_results.append(PostHocPair(
                    group1=group_names[i],
                    group2=group_names[j],
                    mean_diff=round(mean_diff, 4),
                    se=round(se, 4),
                    statistic=round(float(t), 4),
                    p_value=round(float(p_pair), 6),
                    p_adjusted=round(p_adj, 6),
                    ci_lower=round(mean_diff - t_crit * se, 4),
                    ci_upper=round(mean_diff + t_crit * se, 4),
                    significant=p_adj <= alpha,
                ))

    # Chart: boxplot data
    box_data = []
    for g_name, g_series in group_data.items():
        q1 = float(g_series.quantile(0.25))
        q3 = float(g_series.quantile(0.75))
        box_data.append({
            "group": g_name,
            "min": float(g_series.min()),
            "q1": q1,
            "median": float(g_series.median()),
            "q3": q3,
            "max": float(g_series.max()),
            "mean": float(g_series.mean()),
        })

    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type="one_way_anova",
        title=f"One-Way ANOVA — {dependent} by {grouping}",
        variables={"dependent": dependent, "grouping": grouping, "groups": groups_list},
        assumptions=assumptions,
        primary=PrimaryResult(
            statistic_name="F",
            statistic_value=round(float(f_stat), 4),
            df=float(df_between),
            df2=float(df_within),
            p_value=round(float(p_value), 6),
            p_value_formatted=format_p(float(p_value)),
            significance=sig,
            alpha=alpha,
        ),
        descriptives=descriptives,
        effect_size=effect,
        post_hoc=post_hoc_results,
        charts=[ChartData(
            chart_type="boxplot",
            data=box_data,
            config={"title": f"{dependent} by {grouping}", "xLabel": grouping, "yLabel": dependent},
        )],
        interpretation=interpret_anova(
            float(f_stat), float(df_between), float(df_within), float(p_value),
            sig, effect.value, effect.interpretation, dependent,
        ),
        warnings=warnings,
        metadata={
            "n_total": len(df), "missing_excluded": n_dropped,
            "library": "scipy.stats", "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
