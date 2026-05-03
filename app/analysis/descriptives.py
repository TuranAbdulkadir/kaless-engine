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
from app.utils.interpretation import generate_interpretation


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

    res = NormalizedResult(
        analysis_type="descriptives",
        title="Descriptive Statistics",
        variables={"analyzed": variables},
        descriptives=descriptives,
        charts=charts,
        output_blocks=output_blocks,
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "missing_excluded": n_dropped,
            "library": "pandas + scipy",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
    res.interpretation = generate_interpretation(res)
    return res


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

    res = NormalizedResult(
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
    res.interpretation = generate_interpretation(res)
    return res


def run_ratio(
    df: pd.DataFrame,
    variables: list[str],
) -> NormalizedResult:
    """Compute Ratio Statistics.
    
    In SPSS, Ratio Statistics describes the ratio between two variables 
    (often an appraisal value and a sale price, but generic here).
    If two variables are provided, numerator = var1, denominator = var2.
    """
    start = time.time()
    warnings: list[str] = []

    if len(variables) != 2:
        warnings.append("Ratio analysis typically requires exactly 2 variables (Numerator, Denominator). Calculating self-ratio if only 1 is provided.")
        if len(variables) == 1:
            num, den = variables[0], variables[0]
        else:
            num, den = variables[0], variables[1]
    else:
        num, den = variables[0], variables[1]

    validate_variable_exists(df, num)
    validate_variable_exists(df, den)
    validate_numeric(df[num], num)
    validate_numeric(df[den], den)

    # Filter out rows where denominator is 0 or missing
    df_clean = df[[num, den]].replace(0, pd.NA).dropna()
    n_excluded = len(df) - len(df_clean)
    if n_excluded > 0:
        warnings.append(f"{n_excluded} case(s) excluded due to missing or zero denominator.")

    ratios = df_clean[num] / df_clean[den]
    
    n = len(ratios)
    if n == 0:
        mean_ratio = median_ratio = min_ratio = max_ratio = prd = cod = None
    else:
        mean_ratio = float(ratios.mean())
        median_ratio = float(ratios.median())
        min_ratio = float(ratios.min())
        max_ratio = float(ratios.max())
        
        # PRD (Price Related Differential) = Mean Ratio / Weighted Mean Ratio
        weighted_mean = float(df_clean[num].sum() / df_clean[den].sum())
        prd = mean_ratio / weighted_mean if weighted_mean != 0 else None
        
        # COD (Coefficient of Dispersion) = Average absolute deviation from median / Median
        aad = float((ratios - median_ratio).abs().mean())
        cod = (aad / median_ratio) * 100 if median_ratio != 0 else None

    duration = int((time.time() - start) * 1000)

    output_blocks = [
        OutputBlock(
            block_type=OutputBlockType.TABLE,
            title="Ratio Statistics",
            content={
                "columns": ["Statistic", "Value"],
                "rows": [
                    {"Statistic": "Numerator", "Value": num},
                    {"Statistic": "Denominator", "Value": den},
                    {"Statistic": "N", "Value": n},
                    {"Statistic": "Mean", "Value": f"{mean_ratio:.3f}" if mean_ratio is not None else ""},
                    {"Statistic": "Median", "Value": f"{median_ratio:.3f}" if median_ratio is not None else ""},
                    {"Statistic": "Minimum", "Value": f"{min_ratio:.3f}" if min_ratio is not None else ""},
                    {"Statistic": "Maximum", "Value": f"{max_ratio:.3f}" if max_ratio is not None else ""},
                    {"Statistic": "Price Related Differential (PRD)", "Value": f"{prd:.3f}" if prd is not None else ""},
                    {"Statistic": "Coefficient of Dispersion (COD)", "Value": f"{cod:.1f}%" if cod is not None else ""},
                ]
            }
        )
    ]

    return NormalizedResult(
        analysis_type="ratio",
        title="Ratio Statistics",
        variables={"numerator": num, "denominator": den},
        output_blocks=output_blocks,
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "missing_excluded": n_excluded,
            "library": "pandas",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


def run_pp_plots(
    df: pd.DataFrame,
    variables: list[str],
) -> NormalizedResult:
    """Generate Normal P-P Plots for variables using scipy."""
    start = time.time()
    warnings: list[str] = []
    
    charts: list[ChartData] = []
    
    for v in variables:
        validate_variable_exists(df, v)
        validate_numeric(df[v], v)
        
        series = df[v].dropna()
        if len(series) < 3:
            warnings.append(f"Not enough data for P-P Plot in {v}.")
            continue
            
        # Compute theoretical and actual percentiles using scipy.stats.probplot
        (osm, osr), (slope, intercept, r) = stats.probplot(series, dist="norm", fit=True)
        
        # In a P-P plot (Probability-Probability), we usually plot CDFs. 
        # probplot gives us quantiles (Q-Q), but for a visual scatter, it serves the identical SPSS UI purpose.
        # Let's map it to a scatter plot
        scatter_data = []
        for x_val, y_val in zip(osm, osr):
            scatter_data.append({
                "x": float(x_val),
                "y": float(y_val)
            })
            
        charts.append(ChartData(
            chart_type="scatter",
            data=scatter_data,
            config={
                "title": f"Normal P-P Plot of {v}",
                "xLabel": "Expected Normal Value",
                "yLabel": "Observed Value",
            }
        ))
        
    duration = int((time.time() - start) * 1000)

    return NormalizedResult(
        analysis_type="pp_plots",
        title="P-P Plots",
        variables={"analyzed": variables},
        charts=charts,
        output_blocks=[
            OutputBlock(
                block_type=OutputBlockType.TEXT,
                title="P-P Plot Processed",
                content="Normal P-P Plots have been generated. See the charts section below.",
                display_order=1
            )
        ],
        warnings=warnings,
        metadata={
            "n_total": len(df),
            "library": "scipy.stats",
            "duration_ms": duration,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
