"""KALESS Engine — Chi-Square Module.

Implements chi-square test of independence.
Library: scipy.stats
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from app.core.preprocessing import (
    validate_variable_exists, validate_categorical, validate_min_n,
)
from app.core.assumptions import check_independence_expected_freq, build_assumptions_block
from app.core.effect_sizes import cramers_v
from app.core.interpretation import determine_significance, interpret_chi_square, format_p
from app.schemas.results import (
    NormalizedResult, PrimaryResult, ChartData, OutputBlock, OutputBlockType
)


def run_chi_square_independence(
    df: pd.DataFrame,
    variable1: str | list[str] = None,
    variable2: str | list[str] = None,
    rows: str | list[str] = None,
    columns: str | list[str] = None,
    alpha: float = 0.05,
) -> NormalizedResult:
    """Chi-square test of independence. Handles both variable1/2 and rows/columns naming."""
    # Resolve aliases
    v1 = variable1 if variable1 is not None else rows
    v2 = variable2 if variable2 is not None else columns
    
    # Handle list-wrapped single strings from frontend
    v1 = v1[0] if isinstance(v1, list) and len(v1) > 0 else v1
    v2 = v2[0] if isinstance(v2, list) and len(v2) > 0 else v2

    if v1 is None or v2 is None:
        raise ValueError("Chi-square requires two variables (variable1/variable2 or rows/columns).")

    start = time.time()
    warnings: list[str] = []

    validate_variable_exists(df, v1)
    validate_variable_exists(df, v2)
    validate_categorical(df[v1], v1)
    validate_categorical(df[v2], v2)

    # Drop rows with missing in either variable
    clean = df[[v1, v2]].dropna()
    n_dropped = len(df) - len(clean)
    if n_dropped > 0:
        warnings.append(f"{n_dropped} case(s) excluded due to missing values.")

    n = len(clean)
    validate_min_n(n, 5, "chi-square test")

    # Contingency table
    ct = pd.crosstab(clean[v1], clean[v2])

    # Test
    chi2, p_value, dof, expected = stats.chi2_contingency(ct)
    sig = determine_significance(float(p_value), alpha)

    # Assumptions
    expected_check = check_independence_expected_freq(expected)
    assumptions = build_assumptions_block([expected_check])
    if not expected_check.passed:
        warnings.append("Some expected frequencies are below 5. Consider Fisher's exact test for 2×2 tables.")

    # Effect size
    min_dim = min(ct.shape[0], ct.shape[1])
    effect = cramers_v(float(chi2), n, min_dim)

    # Crosstab data for output blocks
    table_rows = []
    for i, row_val in enumerate(ct.index):
        row_dict = {"Row": str(row_val)}
        for j, col_val in enumerate(ct.columns):
            row_dict[str(col_val)] = f"{ct.iloc[i, j]} (Exp: {expected[i, j]:.1f})"
        row_dict["Total"] = int(ct.sum(axis=1).iloc[i])
        table_rows.append(row_dict)
    
    # Add column totals
    col_totals = {"Row": "Total"}
    for j, col_val in enumerate(ct.columns):
        col_totals[str(col_val)] = int(ct.sum(axis=0).iloc[j])
    col_totals["Total"] = n
    table_rows.append(col_totals)

    output_blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title=f"Crosstabulation: {v1} * {v2}",
            content={
                "columns": ["Row"] + [str(c) for c in ct.columns] + ["Total"],
                "rows": table_rows
            }
        ),
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Chi-Square Tests",
            content={
                "columns": ["Test", "Value", "df", "Asymp. Sig. (2-sided)"],
                "rows": [
                    {
                        "Test": "Pearson Chi-Square",
                        "Value": f"{chi2:.3f}",
                        "df": int(dof),
                        "Asymp. Sig. (2-sided)": format_p(p_value)
                    },
                    {
                        "Test": "N of Valid Cases",
                        "Value": str(n),
                        "df": "",
                        "Asymp. Sig. (2-sided)": ""
                    }
                ]
            }
        )
    ]

    # Chart: stacked bar
    chart_data = []
    for row_val in ct.index:
        row_data: dict = {"category": str(row_val)}
        for col_val in ct.columns:
            row_data[str(col_val)] = int(ct.loc[row_val, col_val])
        chart_data.append(row_data)

    duration = int((time.time() - start) * 1000)

    from app.utils.interpretation import generate_interpretation
    res = NormalizedResult(
        analysis_type="chi_square_independence",
        title=f"Chi-Square Test of Independence — {v1} × {v2}",
        variables={"variable1": str(v1), "variable2": str(v2)},
        assumptions=assumptions,
        primary=PrimaryResult(
            statistic_name="χ²",
            statistic_value=round(float(chi2), 4),
            df=float(dof),
            p_value=round(float(p_value), 6),
            p_value_formatted=format_p(float(p_value)),
            significance=sig,
        ),
        output_blocks=output_blocks,
        charts=[
            ChartData(
                title="Observed Frequencies",
                chart_type="bar",
                data=chart_data,
                config={"stacked": True, "xLabel": str(v1), "yLabel": "Frequency"}
            )
        ],
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "missing_excluded": n_dropped,
            "valid_n": n,
            "cramers_v": round(float(effect), 4),
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
    res.interpretation = generate_interpretation(res)
    return res
