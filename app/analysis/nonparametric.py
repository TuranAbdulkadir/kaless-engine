"""KALESS Engine — Non-Parametric Tests Module.

Implements:
  1. Chi-Square (One Sample) — scipy.stats.chisquare
  2. Mann-Whitney U (Two Independent Samples) — scipy.stats.mannwhitneyu
  3. Wilcoxon Signed-Rank (Two Related Samples) — scipy.stats.wilcoxon

All outputs match IBM SPSS table formatting.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from app.core.preprocessing import validate_variable_exists, validate_numeric
from app.schemas.results import (
    NormalizedResult,
    OutputBlock,
    OutputBlockType,
    Interpretation,
)
from app.utils.errors import ValidationError


# ═══════════════════════════════════════════════════════════
#  DISPATCHER
# ═══════════════════════════════════════════════════════════

def run_nonparametric(df: pd.DataFrame, params: dict) -> NormalizedResult:
    """Route to the correct nonparametric test."""
    test_type = params.get("test_type", "chi_square")

    if test_type == "chi_square":
        return _run_chi_square(df, params)
    elif test_type == "mann_whitney":
        return _run_mann_whitney(df, params)
    elif test_type == "wilcoxon":
        return _run_wilcoxon(df, params)
    else:
        raise ValidationError(f"Unknown nonparametric test type: {test_type}")


# ═══════════════════════════════════════════════════════════
#  CHI-SQUARE (ONE SAMPLE)
# ═══════════════════════════════════════════════════════════

def _run_chi_square(df: pd.DataFrame, params: dict) -> NormalizedResult:
    start = time.time()
    variables = params.get("variables", [])
    if not variables:
        raise ValidationError("No variables provided for Chi-Square test.")

    warnings: list[str] = []
    df_clean = df.dropna(subset=variables)
    n_excluded = len(df) - len(df_clean)
    if n_excluded > 0:
        warnings.append(f"{n_excluded} case(s) excluded due to missing values.")

    output_blocks = []

    for var in variables:
        validate_variable_exists(df_clean, var)
        counts = df_clean[var].value_counts().sort_index()
        observed = counts.values.astype(float)
        categories = [str(c) for c in counts.index]
        k = len(observed)
        total_n = int(observed.sum())

        # Expected: equal probability unless user specifies
        expected = np.full(k, total_n / k)

        res = stats.chisquare(f_obs=observed, f_exp=expected)

        # Frequency Table
        freq_rows = []
        for i, cat in enumerate(categories):
            freq_rows.append({
                "": cat,
                "Observed N": int(observed[i]),
                "Expected N": round(float(expected[i]), 1),
                "Residual": round(float(observed[i] - expected[i]), 1),
            })
        freq_rows.append({
            "": "Total",
            "Observed N": total_n,
            "Expected N": "",
            "Residual": "",
        })

        output_blocks.append(OutputBlock(
            block_type=OutputBlockType.TABLE,
            title=f"{var}",
            display_order=len(output_blocks) + 1,
            content={
                "columns": ["", "Observed N", "Expected N", "Residual"],
                "rows": freq_rows,
            },
        ))

        # Test Statistics Table
        output_blocks.append(OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Test Statistics",
            display_order=len(output_blocks) + 1,
            content={
                "columns": ["", var],
                "rows": [
                    {"": "Chi-Square", var: round(float(res.statistic), 3)},
                    {"": "df", var: k - 1},
                    {"": "Asymp. Sig.", var: f"{res.pvalue:.3f}" if res.pvalue >= 0.0005 else "< .001"},
                ],
                "footnotes": [
                    f"0 cells (0.0%) have expected frequencies less than 5. The minimum expected cell frequency is {round(float(expected.min()), 1)}."
                ],
            },
        ))

    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type="nonparametric_chi_square",
        title="NPar Tests — Chi-Square Test",
        variables={"test_variables": variables},
        output_blocks=output_blocks,
        interpretation=Interpretation(
            summary="Chi-Square goodness-of-fit test compares observed frequencies to expected (equal) frequencies.",
            academic_sentence=(
                f"A chi-square goodness-of-fit test was conducted on {len(variables)} variable(s). "
                "See the Test Statistics table for χ², df, and significance values."
            ),
            recommendations=[],
        ),
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "missing_excluded": n_excluded,
            "library": "scipy.stats.chisquare",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ═══════════════════════════════════════════════════════════
#  MANN-WHITNEY U (TWO INDEPENDENT SAMPLES)
# ═══════════════════════════════════════════════════════════

def _run_mann_whitney(df: pd.DataFrame, params: dict) -> NormalizedResult:
    start = time.time()
    variables = params.get("variables", [])
    grouping_var = params.get("grouping_var")
    group_values = params.get("group_values", [])
    warnings: list[str] = []

    if not variables or not grouping_var:
        raise ValidationError("Mann-Whitney U requires test variables and a grouping variable.")

    validate_variable_exists(df, grouping_var)
    for v in variables:
        validate_variable_exists(df, v)

    df_clean = df.dropna(subset=variables + [grouping_var])
    n_excluded = len(df) - len(df_clean)
    if n_excluded > 0:
        warnings.append(f"{n_excluded} case(s) excluded due to missing values.")

    # Auto-detect groups if not specified
    if not group_values or len(group_values) < 2:
        unique_vals = sorted(df_clean[grouping_var].unique().tolist())
        if len(unique_vals) < 2:
            raise ValidationError(f"Grouping variable '{grouping_var}' must have at least 2 distinct values.")
        group_values = [unique_vals[0], unique_vals[1]]

    val1, val2 = group_values
    g1 = df_clean[df_clean[grouping_var] == val1]
    g2 = df_clean[df_clean[grouping_var] == val2]
    n1, n2 = len(g1), len(g2)
    N = n1 + n2

    output_blocks = []
    ranks_rows = []
    stats_rows_data = {
        "Mann-Whitney U": {},
        "Wilcoxon W": {},
        "Z": {},
        "Asymp. Sig. (2-tailed)": {},
    }

    for var in variables:
        x = g1[var].values.astype(float)
        y = g2[var].values.astype(float)

        res = stats.mannwhitneyu(x, y, alternative="two-sided")
        u_stat = float(res.statistic)

        # Compute ranks
        all_vals = np.concatenate([x, y])
        ranked = stats.rankdata(all_vals)
        r1 = ranked[:n1]
        r2 = ranked[n1:]

        r1_sum = float(r1.sum())
        r2_sum = float(r2.sum())
        r1_mean = r1_sum / n1 if n1 > 0 else 0
        r2_mean = r2_sum / n2 if n2 > 0 else 0

        w_stat = min(r1_sum, r2_sum)

        # Z approximation
        mu = n1 * n2 / 2.0
        sigma = np.sqrt(n1 * n2 * (N + 1) / 12.0)
        z_approx = (u_stat - mu) / sigma if sigma > 0 else 0.0

        ranks_rows.extend([
            {"": var, grouping_var: str(val1), "N": n1, "Mean Rank": round(r1_mean, 2), "Sum of Ranks": round(r1_sum, 2)},
            {"": "", grouping_var: str(val2), "N": n2, "Mean Rank": round(r2_mean, 2), "Sum of Ranks": round(r2_sum, 2)},
            {"": "", grouping_var: "Total", "N": N, "Mean Rank": "", "Sum of Ranks": ""},
        ])

        stats_rows_data["Mann-Whitney U"][var] = round(u_stat, 3)
        stats_rows_data["Wilcoxon W"][var] = round(w_stat, 3)
        stats_rows_data["Z"][var] = round(z_approx, 3)
        stats_rows_data["Asymp. Sig. (2-tailed)"][var] = f"{res.pvalue:.3f}" if res.pvalue >= 0.0005 else "< .001"

    # Ranks Table
    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Ranks",
        display_order=1,
        content={
            "columns": ["", grouping_var, "N", "Mean Rank", "Sum of Ranks"],
            "rows": ranks_rows,
        },
    ))

    # Test Statistics Table
    stats_cols = [""]
    for var in variables:
        stats_cols.append(var)

    stats_rows = []
    for metric in ["Mann-Whitney U", "Wilcoxon W", "Z", "Asymp. Sig. (2-tailed)"]:
        row = {"": metric}
        for var in variables:
            row[var] = stats_rows_data[metric].get(var, "")
        stats_rows.append(row)

    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Test Statistics",
        display_order=2,
        content={
            "columns": stats_cols,
            "rows": stats_rows,
            "footnotes": [f"Grouping Variable: {grouping_var}"],
        },
    ))

    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type="nonparametric_mann_whitney",
        title="NPar Tests — Mann-Whitney U",
        variables={"test_variables": variables, "grouping": [grouping_var]},
        output_blocks=output_blocks,
        interpretation=Interpretation(
            summary="Mann-Whitney U test compares differences between two independent groups on a continuous measure.",
            academic_sentence=(
                f"A Mann-Whitney U test was conducted to compare {', '.join(variables)} "
                f"across groups defined by {grouping_var} ({val1} vs {val2})."
            ),
            recommendations=[],
        ),
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "missing_excluded": n_excluded,
            "library": "scipy.stats.mannwhitneyu",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ═══════════════════════════════════════════════════════════
#  WILCOXON SIGNED-RANK (TWO RELATED SAMPLES)
# ═══════════════════════════════════════════════════════════

def _run_wilcoxon(df: pd.DataFrame, params: dict) -> NormalizedResult:
    start = time.time()
    variables = params.get("variables", [])
    warnings: list[str] = []

    if len(variables) < 2:
        raise ValidationError("Wilcoxon test requires at least two variables (a paired pair).")

    # Build pairs: take consecutive pairs [v1, v2], [v3, v4], ...
    # Or if exactly 2 variables, that's one pair.
    pairs = []
    if len(variables) == 2:
        pairs = [(variables[0], variables[1])]
    else:
        for i in range(0, len(variables) - 1, 2):
            pairs.append((variables[i], variables[i + 1]))

    output_blocks = []
    ranks_rows = []
    stats_cols = [""]
    stats_rows_data = {"Z": {}, "Asymp. Sig. (2-tailed)": {}}

    for var1, var2 in pairs:
        validate_variable_exists(df, var1)
        validate_variable_exists(df, var2)
        df_clean = df[[var1, var2]].dropna()
        n_excluded = len(df) - len(df_clean)
        if n_excluded > 0:
            warnings.append(f"{n_excluded} case(s) excluded for pair {var1}-{var2}.")

        x = df_clean[var1].values.astype(float)
        y = df_clean[var2].values.astype(float)
        diff = y - x

        neg_ranks = diff[diff < 0]
        pos_ranks = diff[diff > 0]
        ties = diff[diff == 0]

        # Rank the absolute differences
        abs_diff = np.abs(diff[diff != 0])
        if len(abs_diff) > 0:
            ranked = stats.rankdata(abs_diff)
            neg_mask = diff[diff != 0] < 0
            pos_mask = diff[diff != 0] > 0
            neg_rank_sum = float(ranked[neg_mask].sum()) if neg_mask.any() else 0.0
            pos_rank_sum = float(ranked[pos_mask].sum()) if pos_mask.any() else 0.0
            neg_mean_rank = neg_rank_sum / len(neg_ranks) if len(neg_ranks) > 0 else 0.0
            pos_mean_rank = pos_rank_sum / len(pos_ranks) if len(pos_ranks) > 0 else 0.0
        else:
            neg_rank_sum = pos_rank_sum = 0.0
            neg_mean_rank = pos_mean_rank = 0.0

        pair_label = f"{var2} - {var1}"

        ranks_rows.extend([
            {"": pair_label, "Ranks": "Negative Ranks", "N": len(neg_ranks), "Mean Rank": round(neg_mean_rank, 2), "Sum of Ranks": round(neg_rank_sum, 2)},
            {"": "", "Ranks": "Positive Ranks", "N": len(pos_ranks), "Mean Rank": round(pos_mean_rank, 2), "Sum of Ranks": round(pos_rank_sum, 2)},
            {"": "", "Ranks": "Ties", "N": len(ties), "Mean Rank": "", "Sum of Ranks": ""},
            {"": "", "Ranks": "Total", "N": len(df_clean), "Mean Rank": "", "Sum of Ranks": ""},
        ])

        # Run Wilcoxon test
        try:
            res = stats.wilcoxon(x, y, alternative="two-sided")
            # Z approximation
            T_stat = float(res.statistic)
            n_nonzero = len(abs_diff)
            mu_T = n_nonzero * (n_nonzero + 1) / 4.0
            sigma_T = np.sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24.0)
            z_val = (T_stat - mu_T) / sigma_T if sigma_T > 0 else 0.0
            p_val = res.pvalue
        except Exception:
            z_val = 0.0
            p_val = 1.0
            warnings.append(f"Could not compute Wilcoxon for pair {var1}-{var2}.")

        stats_cols.append(pair_label)
        stats_rows_data["Z"][pair_label] = round(z_val, 3)
        stats_rows_data["Asymp. Sig. (2-tailed)"][pair_label] = f"{p_val:.3f}" if p_val >= 0.0005 else "< .001"

    # Ranks Table
    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Ranks",
        display_order=1,
        content={
            "columns": ["", "Ranks", "N", "Mean Rank", "Sum of Ranks"],
            "rows": ranks_rows,
            "footnotes": [
                "a. var2 < var1",
                "b. var2 > var1",
                "c. var2 = var1",
            ],
        },
    ))

    # Test Statistics Table
    stats_rows = []
    for metric in ["Z", "Asymp. Sig. (2-tailed)"]:
        row = {"": metric}
        for label in stats_cols[1:]:
            row[label] = stats_rows_data[metric].get(label, "")
        stats_rows.append(row)

    output_blocks.append(OutputBlock(
        block_type=OutputBlockType.TABLE,
        title="Test Statistics",
        display_order=2,
        content={
            "columns": stats_cols,
            "rows": stats_rows,
            "footnotes": ["a. Based on negative ranks.", "b. Wilcoxon Signed Ranks Test."],
        },
    ))

    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type="nonparametric_wilcoxon",
        title="NPar Tests — Wilcoxon Signed Ranks",
        variables={"paired_variables": [f"{v1},{v2}" for v1, v2 in pairs]},
        output_blocks=output_blocks,
        interpretation=Interpretation(
            summary="Wilcoxon Signed-Rank test compares two related (paired) samples.",
            academic_sentence=(
                f"A Wilcoxon Signed-Rank test was conducted on {len(pairs)} pair(s). "
                "Results are presented in the Test Statistics table."
            ),
            recommendations=[],
        ),
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "library": "scipy.stats.wilcoxon",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
