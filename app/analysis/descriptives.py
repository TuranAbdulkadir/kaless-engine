"""KALESS Engine — Descriptive Statistics Module.

Implements frequencies, descriptives, explore, ratio, P-P/Q-Q plots, and crosstabs.
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
    Interpretation, OutputBlock, OutputBlockType, PrimaryResult
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
    # Add charts to output blocks for rendering
    for chart in res.charts:
        res.output_blocks.append(OutputBlock(
            block_type=OutputBlockType.CHART,
            title=chart.config.get("title", "Chart"),
            content=chart.dict()
        ))
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
    # Add charts to output blocks
    for chart in res.charts:
        res.output_blocks.append(OutputBlock(
            block_type=OutputBlockType.CHART,
            title=chart.config.get("title", "Frequency Chart"),
            content=chart.dict()
        ))
    return res


def run_ratio(
    df: pd.DataFrame,
    variables: list[str],
) -> NormalizedResult:
    """Compute Ratio Statistics."""
    start = time.time()
    warnings: list[str] = []

    if len(variables) < 2:
        raise ValueError("Ratio analysis requires exactly 2 variables (Numerator, Denominator).")
    
    num, den = variables[0], variables[1]

    validate_variable_exists(df, num)
    validate_variable_exists(df, den)
    validate_numeric(df[num], num)
    validate_numeric(df[den], den)

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
        weighted_mean = float(df_clean[num].sum() / df_clean[den].sum())
        prd = mean_ratio / weighted_mean if weighted_mean != 0 else None
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

    res = NormalizedResult(
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
    res.interpretation = generate_interpretation(res)
    return res


def run_pp_plots(
    df: pd.DataFrame,
    variables: list[str],
) -> NormalizedResult:
    """Generate Normal P-P Plots."""
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
        (osm, osr), (slope, intercept, r) = stats.probplot(series, dist="norm", fit=True)
        scatter_data = [{"x": float(x), "y": float(y)} for x, y in zip(osm, osr)]
        charts.append(ChartData(
            chart_type="scatter",
            data=scatter_data,
            config={"title": f"Normal P-P Plot of {v}", "xLabel": "Expected Normal", "yLabel": "Observed"}
        ))
        
    res = NormalizedResult(
        analysis_type="pp_plots",
        title="P-P Plots",
        variables={"analyzed": variables},
        charts=charts,
        output_blocks=[
            OutputBlock(
                block_type=OutputBlockType.TEXT, 
                title="P-P Plot Processed", 
                content={"text": "Normal P-P Plots generated. The plots visualize the cumulative distribution of the data against a theoretical normal distribution."}
            )
        ],
        metadata={"library": "scipy.stats", "timestamp": datetime.utcnow().isoformat()}
    )
    # Add charts to output blocks for rendering
    for chart in charts:
        res.output_blocks.append(OutputBlock(
            block_type=OutputBlockType.CHART,
            title=chart.config.get("title", "P-P Plot"),
            content=chart.dict()
        ))
    res.interpretation = generate_interpretation(res)
    return res


def run_qq_plots(
    df: pd.DataFrame,
    variables: list[str],
) -> NormalizedResult:
    """Generate Normal Q-Q Plots."""
    start = time.time()
    charts: list[ChartData] = []
    for v in variables:
        validate_variable_exists(df, v)
        validate_numeric(df[v], v)
        series = df[v].dropna()
        if len(series) < 3: continue
        (osm, osr), (slope, intercept, r) = stats.probplot(series, dist="norm", fit=True)
        scatter_data = [{"x": float(x), "y": float(y)} for x, y in zip(osm, osr)]
        charts.append(ChartData(
            chart_type="scatter",
            data=scatter_data,
            config={"title": f"Normal Q-Q Plot of {v}", "xLabel": "Theoretical Quantiles", "yLabel": "Sample Quantiles"}
        ))
    res = NormalizedResult(
        analysis_type="qq_plots",
        title="Q-Q Plots",
        variables={"analyzed": variables},
        charts=charts,
        output_blocks=[
            OutputBlock(
                block_type=OutputBlockType.TEXT, 
                title="Q-Q Plot Processed", 
                content={"text": "Normal Q-Q Plots generated. These plots compare the quantiles of the data distribution with the quantiles of a normal distribution."}
            )
        ],
        metadata={"library": "scipy.stats", "timestamp": datetime.utcnow().isoformat()}
    )
    # Add charts to output blocks for rendering
    for chart in charts:
        res.output_blocks.append(OutputBlock(
            block_type=OutputBlockType.CHART,
            title=chart.config.get("title", "Q-Q Plot"),
            content=chart.dict()
        ))
    res.interpretation = generate_interpretation(res)
    return res


def run_crosstabs(
    df: pd.DataFrame,
    rows: str | list[str],
    columns: str | list[str],
) -> NormalizedResult:
    """Generate Crosstabulation table."""
    from app.analysis.chi_square import run_chi_square_independence
    
    # Handle list inputs from frontend
    row_var = rows[0] if isinstance(rows, list) else rows
    col_var = columns[0] if isinstance(columns, list) else columns
    
    return run_chi_square_independence(df, row_var, col_var)


def run_explore(
    df: pd.DataFrame,
    dependent: list[str],
    grouping: str,
    alpha: float = 0.05,
) -> NormalizedResult:
    """Detailed exploration of dependents by grouping factor."""
    start = time.time()
    warnings: list[str] = []
    for v in dependent:
        validate_variable_exists(df, v)
        validate_numeric(df[v], v)
    validate_variable_exists(df, grouping)

    cleaned, n_dropped = drop_missing_listwise(df, dependent + [grouping])
    if n_dropped > 0: warnings.append(f"{n_dropped} case(s) excluded.")

    output_blocks = []
    descriptives = []
    for dep_var in dependent:
        groups = cleaned.groupby(grouping)[dep_var]
        table_rows = []
        for name, group in groups:
            desc_stats = compute_descriptive(group, name=str(name))
            descriptives.append(GroupDescriptive(**desc_stats))
            table_rows.append({
                "Group": str(name), "N": desc_stats["n"], "Mean": round(desc_stats["mean"], 3),
                "Std. Deviation": round(desc_stats["sd"], 3), "Std. Error": round(desc_stats["se"], 3),
                "95% CI Lower": round(desc_stats["mean"] - 1.96 * desc_stats["se"], 3),
                "95% CI Upper": round(desc_stats["mean"] + 1.96 * desc_stats["se"], 3),
                "Min": round(desc_stats["min"], 3), "Max": round(desc_stats["max"], 3)
            })
        output_blocks.append(OutputBlock(
            block_type=OutputBlockType.TABLE,
            title=f"Explore: {dep_var} by {grouping}",
            content={"columns": ["Group", "N", "Mean", "Std. Deviation", "Std. Error", "95% CI Lower", "95% CI Upper", "Min", "Max"], "rows": table_rows}
        ))

    res = NormalizedResult(
        analysis_type="explore",
        title="Explore Analysis",
        variables={"dependent": dependent, "factor": grouping},
        descriptives=descriptives,
        output_blocks=output_blocks,
        warnings=warnings,
        metadata={"n_total": len(df), "duration_ms": int((time.time() - start) * 1000), "timestamp": datetime.utcnow().isoformat()},
    )
    res.interpretation = generate_interpretation(res)
    return res
