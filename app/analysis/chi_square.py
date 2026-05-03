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
    NormalizedResult, PrimaryResult, ChartData,
)


def run_chi_square_independence(
    df: pd.DataFrame,
    variable1: str,
    variable2: str,
    alpha: float = 0.05,
) -> NormalizedResult:
    """Chi-square test of independence between two categorical variables."""
    start = time.time()
    warnings: list[str] = []

    validate_variable_exists(df, variable1)
    validate_variable_exists(df, variable2)
    validate_categorical(df[variable1], variable1)
    validate_categorical(df[variable2], variable2)

    # Drop rows with missing in either variable
    clean = df[[variable1, variable2]].dropna()
    n_dropped = len(df) - len(clean)
    if n_dropped > 0:
        warnings.append(f"{n_dropped} case(s) excluded due to missing values.")

    n = len(clean)
    validate_min_n(n, 5, "chi-square test")

    # Contingency table
    ct = pd.crosstab(clean[variable1], clean[variable2])

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

    # Crosstab data
    observed_dict = {
        "rows": ct.index.tolist(),
        "columns": ct.columns.tolist(),
        "observed": ct.values.tolist(),
        "expected": expected.tolist(),
        "row_totals": ct.sum(axis=1).tolist(),
        "column_totals": ct.sum(axis=0).tolist(),
        "grand_total": n,
    }

    # Stringify for JSON safety
    observed_dict["rows"] = [str(r) for r in observed_dict["rows"]]
    observed_dict["columns"] = [str(c) for c in observed_dict["columns"]]

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
        title=f"Chi-Square Test of Independence — {variable1} × {variable2}",
        variables={"variable1": variable1, "variable2": variable2},
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
                title="Observed Frequencies (Heatmap data)",
                chart_type="heatmap",
                data=chart_data,
            )
        ],
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "missing_excluded": excluded_n,
            "valid_n": valid_n,
            "cramers_v": round(float(cv), 4),
            "phi": round(float(phi), 4),
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
    res.interpretation = generate_interpretation(res)
    return res
