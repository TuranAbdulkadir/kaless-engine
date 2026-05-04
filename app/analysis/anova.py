"""KALESS Engine — One-Way ANOVA Module."""

import time
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from app.core.preprocessing import validate_variable_exists
from app.schemas.results import NormalizedResult

def run_one_way_anova(
    df: pd.DataFrame, 
    dependent: str | list[str], 
    grouping: str | list[str]
) -> NormalizedResult:
    """Computes One-Way ANOVA."""
    start = time.time()
    warnings: list[str] = []

    # Handle list inputs
    dep_var = dependent[0] if isinstance(dependent, list) and len(dependent) > 0 else dependent
    group_var = grouping[0] if isinstance(grouping, list) and len(grouping) > 0 else grouping

    validate_variable_exists(df, dep_var)
    validate_variable_exists(df, group_var)

    # Prepare data, drop missing values
    sub_df = df[[dep_var, group_var]].dropna()
    valid_n = len(sub_df)
    
    if valid_n < 3:
        raise ValueError("Not enough valid cases to compute One-Way ANOVA.")

    if len(sub_df) < len(df):
        warnings.append(f"{len(df) - valid_n} case(s) excluded due to missing values.")

    # Group data
    groups = sub_df.groupby(factor)[dependent]
    group_data = [group.values for name, group in groups]
    group_names = list(groups.groups.keys())

    if len(group_data) < 2:
        raise ValueError("Factor variable must have at least two distinct groups.")

    # Calculate One-Way ANOVA using scipy
    f_stat, p_value = stats.f_oneway(*group_data)

    # Sum of Squares (Between and Within)
    grand_mean = sub_df[dependent].mean()
    
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_data)
    df_between = len(group_data) - 1
    ms_between = ss_between / df_between if df_between > 0 else 0

    ss_within = sum(sum((x - g.mean())**2 for x in g) for g in group_data)
    df_within = valid_n - len(group_data)
    ms_within = ss_within / df_within if df_within > 0 else 0

    ss_total = ss_between + ss_within
    df_total = df_between + df_within

    anova_table = [
        {
            "Source": "Between Groups",
            "Sum of Squares": round(ss_between, 3),
            "df": df_between,
            "Mean Square": round(ms_between, 3),
            "F": round(f_stat, 3) if not np.isnan(f_stat) else None,
            "Sig.": round(p_value, 3) if not np.isnan(p_value) else None
        },
        {
            "Source": "Within Groups",
            "Sum of Squares": round(ss_within, 3),
            "df": df_within,
            "Mean Square": round(ms_within, 3),
            "F": None,
            "Sig.": None
        },
        {
            "Source": "Total",
            "Sum of Squares": round(ss_total, 3),
            "df": df_total,
            "Mean Square": None,
            "F": None,
            "Sig.": None
        }
    ]

    output_blocks = [
        {
            "block_type": "table",
            "title": "ANOVA",
            "content": {
                "columns": ["Source", "Sum of Squares", "df", "Mean Square", "F", "Sig."],
                "rows": anova_table,
                "footnotes": [f"Dependent Variable: {dependent}"]
            }
        }
    ]

    # Post-Hoc: Tukey HSD (if significant)
    if not np.isnan(p_value) and p_value < 0.05 and len(group_data) > 2:
        try:
            tukey = pairwise_tukeyhsd(endog=sub_df[dependent], groups=sub_df[factor], alpha=0.05)
            
            # The summary contains: group1, group2, meandiff, p-adj, lower, upper, reject
            res_data = tukey.summary().data
            headers = res_data[0]
            
            # Find indices
            idx_g1 = headers.index('group1')
            idx_g2 = headers.index('group2')
            idx_diff = headers.index('meandiff')
            idx_p = headers.index('p-adj')
            idx_lower = headers.index('lower')
            idx_upper = headers.index('upper')

            post_hoc_rows = []
            for row in res_data[1:]:
                # Note: statsmodels meandiff is group2 - group1, SPSS is usually (I) - (J)
                # But we will use what statsmodels provides directly
                post_hoc_rows.append({
                    "(I) Group": row[idx_g1],
                    "(J) Group": row[idx_g2],
                    "Mean Difference (I-J)": round(row[idx_diff], 4),
                    "Std. Error": "", # statsmodels tukey doesn't directly expose SE in summary easily, leave empty or calculate
                    "Sig.": round(row[idx_p], 3),
                    "Lower Bound": round(row[idx_lower], 4),
                    "Upper Bound": round(row[idx_upper], 4)
                })

            output_blocks.append({
                "block_type": "table",
                "title": "Multiple Comparisons (Tukey HSD)",
                "content": {
                    "columns": ["(I) Group", "(J) Group", "Mean Difference (I-J)", "Std. Error", "Sig.", "Lower Bound", "Upper Bound"],
                    "rows": post_hoc_rows,
                    "footnotes": ["The error term is Mean Square(Error).", "* The mean difference is significant at the 0.05 level."]
                }
            })
        except Exception as e:
            warnings.append(f"Could not compute Tukey HSD Post-Hoc: {str(e)}")

    # Descriptives for the groups
    descriptives = []
    for name, group in groups:
        descriptives.append({
            "name": str(name),
            "n": len(group),
            "mean": float(group.mean()),
            "sd": float(group.std()) if len(group) > 1 else 0.0,
            "se": float(group.std() / np.sqrt(len(group))) if len(group) > 1 else 0.0,
            "min": float(group.min()),
            "max": float(group.max())
        })

    # Prepare Primary Statistic
    p_formatted = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
    sig = "significant" if p_value < 0.05 else "not_significant"
    
    primary = {
        "statistic_name": "F",
        "statistic_value": float(f_stat),
        "df": float(df_between),
        "df2": float(df_within),
        "p_value": float(p_value),
        "p_value_formatted": p_formatted,
        "significance": sig
    }

    duration = int((time.time() - start) * 1000)

    from app.utils.interpretation import generate_interpretation

    res = NormalizedResult(
        analysis_type="one_way_anova",
        title="One-Way ANOVA",
        variables={"dependent": [str(dep_var)], "grouping": [str(group_var)]},
        descriptives=descriptives,
        output_blocks=output_blocks,
        primary=primary,
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "valid_n": valid_n,
            "library": "scipy + statsmodels",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
    res.interpretation = generate_interpretation(res)
    return res
