"""KALESS Engine — Descriptive Statistics Module.

Implements frequencies, descriptives, and explore analyses.
Library: pandas + scipy
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
from scipy import stats

from app.core.preprocessing import (
    validate_variable_exists,
    validate_numeric,
    drop_missing_listwise,
    compute_descriptive,
)
from app.schemas.results import (
    NormalizedResult, GroupDescriptive, ChartData,
    Interpretation, OutputBlock, OutputBlockType
)


def run_descriptives(
    df: pd.DataFrame,
    variables: list[str],
    alpha: float = 0.05,
) -> NormalizedResult:
    """Compute descriptive statistics for one or more numeric variables."""
    start = time.time()
    warnings: list[str] = []

    for v in variables:
        validate_variable_exists(df, v)
        validate_numeric(df[v], v)

    cleaned, n_dropped = drop_missing_listwise(df, variables)
    if n_dropped > 0:
        warnings.append(f"{n_dropped} case(s) excluded due to missing values.")

    descriptives: list[GroupDescriptive] = []
    charts: list[ChartData] = []

    for v in variables:
        series = cleaned[v]
        desc = compute_descriptive(series, name=v)
        descriptives.append(GroupDescriptive(**desc))

        # Histogram data
        if len(series) > 0:
            hist_counts, bin_edges = pd.cut(series, bins=min(20, max(5, len(series) // 10)), retbins=True)
            hist_data = []
            value_counts = hist_counts.value_counts().sort_index()
            for interval, count in value_counts.items():
                hist_data.append({
                    "bin": f"{interval.left:.1f}-{interval.right:.1f}",
                    "count": int(count),
                    "midpoint": float((interval.left + interval.right) / 2),
                })
            charts.append(ChartData(
                chart_type="histogram",
                data=hist_data,
                config={"title": f"Distribution of {v}", "xLabel": v, "yLabel": "Frequency"},
            ))

    duration = int((time.time() - start) * 1000)

    summary_parts = []
    for d in descriptives:
        if d.mean is not None and d.sd is not None:
            summary_parts.append(f"{d.name}: M = {d.mean:.3f}, SD = {d.sd:.3f}, N = {d.n}")

    output_blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Descriptive Statistics",
            content={
                "columns": ["Variable", "N", "Mean", "Std. Deviation", "Minimum", "Maximum"],
                "rows": [
                    {
                        "Variable": d.name,
                        "N": d.n,
                        "Mean": f"{d.mean:.3f}" if d.mean is not None else "",
                        "Std. Deviation": f"{d.sd:.3f}" if d.sd is not None else "",
                        "Minimum": f"{d.min:.3f}" if d.min is not None else "",
                        "Maximum": f"{d.max:.3f}" if d.max is not None else ""
                    } for d in descriptives
                ]
            }
        )
    ]

    return NormalizedResult(
        analysis_type="descriptives",
        title="Descriptive Statistics",
        variables={"analyzed": variables},
        descriptives=descriptives,
        charts=charts,
        output_blocks=output_blocks,
        warnings=warnings,
        interpretation=Interpretation(
            summary="; ".join(summary_parts) if summary_parts else "Descriptive statistics computed.",
            academic_sentence="Descriptive statistics are reported in the table above.",
            recommendations=[],
        ),
        metadata={
            "n_total": len(df),
            "missing_excluded": n_dropped,
            "library": "pandas + scipy",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


def run_frequencies(
    df: pd.DataFrame,
    variable: str,
) -> NormalizedResult:
    """Compute frequency distribution for a variable."""
    start = time.time()
    warnings: list[str] = []

    validate_variable_exists(df, variable)

    series = df[variable].dropna()
    n_missing = int(df[variable].isna().sum())
    if n_missing > 0:
        warnings.append(f"{n_missing} missing value(s) excluded.")

    freq = series.value_counts().sort_index()
    total = len(series)

    freq_table = []
    cumulative = 0
    for val, count in freq.items():
        cumulative += count
        freq_table.append({
            "value": str(val),
            "frequency": int(count),
            "percent": round(count / total * 100, 1) if total > 0 else 0,
            "cumulative_percent": round(cumulative / total * 100, 1) if total > 0 else 0,
        })

    # Chart data
    chart_data = [{"category": str(val), "count": int(count)} for val, count in freq.items()]

    duration = int((time.time() - start) * 1000)

    output_blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title=f"Frequencies: {variable}",
            content={
                "columns": ["Value", "Frequency", "Percent", "Cumulative Percent"],
                "rows": [
                    {
                        "Value": f["value"],
                        "Frequency": f["frequency"],
                        "Percent": f["percent"],
                        "Cumulative Percent": f["cumulative_percent"]
                    } for f in freq_table
                ]
            }
        )
    ]

    return NormalizedResult(
        analysis_type="frequencies",
        title=f"Frequency Table — {variable}",
        variables={"analyzed": [variable]},
        frequency_table=freq_table,
        descriptives=[GroupDescriptive(name=variable, n=total)],
        charts=[ChartData(
            chart_type="bar",
            data=chart_data,
            config={"title": f"Frequencies of {variable}", "xLabel": variable, "yLabel": "Count"},
        )],
        output_blocks=output_blocks,
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "missing_excluded": n_missing,
            "library": "pandas",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
